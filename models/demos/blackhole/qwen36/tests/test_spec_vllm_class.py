# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Qwen36MTPContractForCausalLM under the vLLM TT plugin (skipped without vllm / vllm_tt_plugin).

Host tests (CPU, fake engine): spec_plan, capabilities, wrapper plumbing.
Device test: drive the class exactly as the plugin runner (#131) calls a model, without a server:
initialize_vllm_model -> allocate_kv_cache_per_layer -> 2-phase warmup -> prefill_forward / decode_forward /
propose_draft_tokens / release_request per request -> release_persistent_capture, with the mamba_state of the GDN state
blocks (mamba_cache_mode "none", tests/spec_block_helpers.py). Output must stay lossless against greedy decoding of
Qwen36ForCausalLM, the class that serves the model without speculation, on its own vLLM blocks (exact up to the first
bf16 near-tie flip).
Runs under TT_METAL_TRACE_ALLOC_TRACKING=1 unchanged.
"""

import functools
import gc
import math
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from loguru import logger

pytest.importorskip("vllm")
pytest.importorskip("vllm_tt_plugin")

pytest.importorskip("vllm_tt_plugin.spec_admission")
pytest.importorskip("vllm_tt_plugin.spec_decode")

from vllm_tt_plugin.spec_admission import resolve_speculative_plan  # noqa: E402
from vllm_tt_plugin.spec_decode import DraftOutput, SpecPlan, SpecReject, VerifyOutput  # noqa: E402

import ttnn  # noqa: E402
from models.common.utility_functions import run_for_blackhole  # noqa: E402
from models.demos.blackhole.qwen36.demo.text_demo import _MESH_SHAPE, _MULTI, DEVICE_PARAMS, _get_prompt  # noqa: E402
from models.demos.blackhole.qwen36.tests.gdn_block_helpers import PluginLayout  # noqa: E402
from models.demos.blackhole.qwen36.tests.spec_block_helpers import SpecBlocks  # noqa: E402
from models.demos.blackhole.qwen36.tests.spec_helpers import (  # noqa: E402
    PAD,
    _assert_matches_ref,
    _prompt_pool,
    _top2_gap,
)
from models.demos.blackhole.qwen36.tests.test_spec_lossless import _N_LAYERS  # noqa: E402
from models.demos.blackhole.qwen36.tt import qwen36_vllm  # noqa: E402
from models.demos.blackhole.qwen36.tt.qwen36_vllm import (  # noqa: E402
    Qwen36ForCausalLM,
    Qwen36MTPContractForCausalLM,
    TT_Qwen3_5ProcessingInfo,
    TT_Qwen3_5SpecProcessingInfo,
)
from models.demos.blackhole.qwen36.tt.spec_engine import (  # noqa: E402
    DraftResult,
    VerifyResult,
    spec_memory_costs,
    spec_table_width,
)

K = 3
MAX_MODEL_LEN = 8192


def _sample(logits, gen, temperature=0.7, top_p=0.95):
    """Host temperature + top-p sample of one [V] logits row."""
    probs = torch.softmax(logits.float() / temperature, dim=-1)
    sp, si = probs.sort(descending=True)
    keep = (sp.cumsum(-1) - sp) < top_p  # always keeps the top token
    sp = sp * keep
    return int(si[torch.multinomial(sp / sp.sum(), 1, generator=gen)])


def _ordinary_greedy(ref, reqs):
    """Greedy references {name: (tokens, top-2 gaps)} of ``reqs`` [(name, prompt, n)] from ``ref``, a
    Qwen36ForCausalLM built with initialize_vllm_model as the plugin builds it, on its own vLLM blocks
    (mamba_cache_mode "none") with traced decode: the computation that serves the model without speculation. Its GDN
    decode is gdn_decode_step, which gdn_spec_step equals bit for bit, so the two classes differ only in the attention
    kernels. Releases the class's traces and KV cache before it returns, so the MTP class allocates after them."""
    model = ref.model[0]
    probe = PluginLayout.for_model(model, num_blocks=1)
    width = math.ceil(MAX_MODEL_LEN / probe.block_size)
    layout = PluginLayout.for_model(model, num_blocks=1 + width + probe.num_groups + 8)
    kv = ref.allocate_kv_cache_per_layer(layout.per_layer_specs)
    for enable_trace in (False, True):
        ref.warmup_model_prefill(enable_trace=enable_trace, kv_cache=kv, can_sample_on_device=False)
        ref.warmup_model_decode(
            enable_trace=enable_trace, kv_cache=kv, max_batch_size=1, num_blocks=width, can_sample_on_device=False
        )
    pages = list(range(1, 1 + width))
    table = torch.tensor([pages], dtype=torch.int32)
    vocab = model.args.vocab_size
    refs = {}
    for name, prompt, n in reqs:
        state = SpecBlocks(layout, 0, "none", free=range(1 + width, layout.num_blocks))
        state.admit(0)
        ref.zero_kv_cache_blocks(kv, pages)  # vLLM's new_block_ids_to_zero for a new request's attention blocks
        P = len(prompt)
        logits, _ = ref.prefill_forward(
            torch.tensor([prompt], dtype=torch.int32),
            table,
            kv,
            prompt_lens=np.array([P]),
            start_pos=np.array([0]),
            empty_slots=[0],
            enable_trace=True,
            mamba_state=state.state({0: state.prefill(0, 0, P)}, 1, intermediate=()),
        )
        row = logits[0, -1, :vocab].float()
        toks, gaps = [int(row.argmax())], [_top2_gap(row)]
        for pos in range(P, P + n - 1):
            out = ref.decode_forward(
                tokens=torch.tensor([[toks[-1]]], dtype=torch.int32),
                start_pos=torch.tensor([pos], dtype=torch.int32),
                page_table=table,
                kv_cache=kv,
                enable_trace=True,
                read_from_device=True,
                reload_inputs=True,
                reload_page_table=False,
                reload_sampling_params=False,
                reset_sampling_state=False,
                mamba_state=state.state({0: state.decode(0, pos, 1)}, 1),
            )
            out = out[0] if isinstance(out, tuple) else out
            row = out[0, -1, :vocab].float()
            toks.append(int(row.argmax()))
            gaps.append(_top2_gap(row))
        refs[name] = (toks, gaps)
        logger.info(f"[vllm_class] reference {name}: {n} tokens from Qwen36ForCausalLM")
    ref.release_persistent_capture()
    model.free_kv_caches()
    return refs


class PluginRunner:
    """Host-side B=1 mirror of the plugin runner's calls into the model class.

    A step is a verify iff it has drafts or the previous commit was > 1 token; otherwise a narrow decode
    (no spec_mode, host-sampled from the returned logits) followed by propose_draft_tokens. A request with
    speculable=False never has its drafts published, so it is always narrow."""

    def __init__(self, model, kv, layout, seed, speculable=True, sample_seed=0):
        self.m, self.kv, self.K = model, kv, K
        self.speculable = speculable
        self.gen = torch.Generator().manual_seed(sample_seed)
        self.bs = layout.block_size
        self.width = math.ceil(MAX_MODEL_LEN / self.bs)
        # Seeded shuffled pool of real blocks; block 0 is vLLM's null block and is never handed out. The GDN state
        # blocks (K + 1 per group, mode "none") come from the same pool as the attention blocks.
        g = torch.Generator().manual_seed(seed)
        free = (torch.randperm(layout.num_blocks - 1, generator=g) + 1).tolist()
        self.state = SpecBlocks(layout, K, "none", free=free[: layout.num_groups * (K + 1)])
        self.free = free[layout.num_groups * (K + 1) :]
        self.blocks = []

    def _table(self, last_pos):
        """[1, width] int32 table covering positions up to last_pos + (1+K) + (K+1) lookahead, zero padded."""
        need = min(self.width, math.ceil((last_pos + (1 + K) + (K + 1)) / self.bs))
        new = []
        while len(self.blocks) < need:
            new.append(self.free.pop())
            self.blocks.append(new[-1])
        if new:
            # vLLM's new_block_ids_to_zero: a new attention block can hold another request's GDN state, whose float32
            # bytes the verify's attention reads as NaN or Inf at the positions it masks.
            self.m.zero_kv_cache_blocks(self.kv, new)
        return torch.tensor([self.blocks + [0] * (self.width - len(self.blocks))], dtype=torch.int32)

    def _pick(self, logits_row):
        return int(logits_row.argmax()) if self.speculable else _sample(logits_row, self.gen)

    def run(self, prompt, max_new, n_policy):
        m, cap, bmax = self.m, self.width * self.bs, self.m.model[0].args.max_batch_size
        P = len(prompt)
        tokens = torch.tensor([prompt], dtype=torch.int32)
        self.state.admit(0)
        logits, _ = m.prefill_forward(
            tokens,
            self._table(P - 1),
            self.kv,
            prompt_lens=torch.tensor([P]).numpy(),
            start_pos=torch.tensor([0]).numpy(),
            empty_slots=[0],
            enable_trace=True,
            mamba_state=self.state.state({0: self.state.prefill(0, 0, P)}, bmax, intermediate=()),
        )
        out = [self._pick(logits[0, -1])]
        steps = []
        proposal = []  # no drafts right after prefill
        acc = 1
        stop = "max_new"
        while len(out) < max_new:
            pos0 = P + len(out) - 1  # position of the last committed token
            if pos0 + 1 + K >= cap:
                stop = "capacity"
                break
            n = min(len(proposal), n_policy(len(steps))) if self.speculable else 0
            drafts = proposal[:n]
            verify = n > 0 or acc > 1
            mamba_state = self.state.state({0: self.state.decode(0, pos0, 1 + n)}, bmax)
            # The plugin also sends one table per layer when the cache has MambaSpec groups.
            per_layer = [self._table(pos0)] * len(self.m.model[0].layer_indices)
            if not verify:
                # Ordinary narrow decode: no spec_mode, host sampling on logits[:, -1, :].
                tt_out = m.decode_forward(
                    tokens=torch.tensor([[out[-1]]], dtype=torch.int32),
                    start_pos=torch.tensor([pos0], dtype=torch.int32),
                    page_table=self._table(pos0),
                    kv_cache=self.kv,
                    enable_trace=True,
                    read_from_device=False,
                    num_valid_drafts=torch.zeros(1, dtype=torch.int32),
                    accepted_counts=torch.ones(1, dtype=torch.int32),
                    reload_inputs=True,
                    reload_page_table=False,
                    reload_sampling_params=False,
                    reset_sampling_state=False,
                    mamba_state=mamba_state,
                    page_tables_per_layer=per_layer,
                )
                assert tuple(tt_out.shape[:2]) == (1, 1), f"narrow logits shape {tuple(tt_out.shape)}"
                new, j, hidden = [self._pick(tt_out[:, -1, :][0])], 0, None
            else:
                tok = torch.full((1, 1 + K), PAD, dtype=torch.int32)
                pos = torch.full((1, 1 + K), PAD, dtype=torch.int32)
                tok[0, : 1 + n] = torch.tensor([out[-1]] + drafts, dtype=torch.int32)
                pos[0, : 1 + n] = torch.arange(pos0, pos0 + 1 + n, dtype=torch.int32)
                res = m.decode_forward(
                    tokens=tok,
                    start_pos=pos,
                    page_table=self._table(pos0),
                    kv_cache=self.kv,
                    enable_trace=True,
                    read_from_device=False,
                    num_valid_drafts=torch.tensor([n], dtype=torch.int32),
                    accepted_counts=torch.tensor([acc], dtype=torch.int32),
                    spec_mode="argmax_ids",
                    reload_inputs=True,
                    reload_page_table=False,
                    reload_sampling_params=False,
                    reset_sampling_state=False,
                    mamba_state=mamba_state,
                    page_tables_per_layer=per_layer,
                )
                assert res.spec_mode == "argmax_ids"
                ids = res.argmax_ids[0].tolist()
                assert all(i == PAD for i in ids[n + 1 :]), f"step {len(steps)}: ids past n={n} not -1: {ids}"
                assert all(i != PAD for i in ids[: n + 1]), f"step {len(steps)}: -1 inside valid ids {ids} (n={n})"
                j = 0
                while j < n and drafts[j] == ids[j]:
                    j += 1
                new, hidden = drafts[:j] + [ids[j]], res.hidden
            new = new[: max_new - len(out)]
            out.extend(new)
            acc = len(new)
            self.state.commit(0, acc)
            committed = torch.full((1, 1 + K), PAD, dtype=torch.int32)
            cpos = torch.full((1, 1 + K), PAD, dtype=torch.int32)
            committed[0, :acc] = torch.tensor(new, dtype=torch.int32)
            cpos[0, :acc] = torch.arange(pos0 + 1, pos0 + 1 + acc, dtype=torch.int32)
            d = m.propose_draft_tokens(K, committed, cpos, torch.tensor([acc], dtype=torch.int32), hidden=hidden)
            assert tuple(d.draft_token_ids.shape) == (1, K)
            nv = int(d.num_valid[0])
            assert 0 <= nv <= K
            # _publish_draft: a non-speculable request's proposal is withheld.
            proposal = [int(t) for t in d.draft_token_ids[0, :nv]] if self.speculable else []
            steps.append(
                {
                    "n": n,
                    "m": j,
                    "num_valid": nv,
                    "verify": verify,
                    "skipped": m._engine().skipped_propose_calls,
                }
            )
        m.release_request(0)
        self.state.release(0)
        return out, {"steps": steps, "stop": stop}


def _log(name, out, st):
    steps = st["steps"]
    committed = sum(s["m"] + 1 for s in steps)
    logger.info(
        f"[vllm_class] {name}: tokens={len(out)} steps={len(steps)} "
        f"mean_committed/step={committed / max(len(steps), 1):.2f} "
        f"verify_steps={sum(s['verify'] for s in steps)} stop={st['stop']}"
    )


@run_for_blackhole()
@pytest.mark.timeout(7200)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("warmup", ["plugin", "lazy"])
def test_spec_vllm_class_runner(mesh_device, monkeypatch, warmup):
    """Plugin call pattern on Qwen36MTPContractForCausalLM: lossless, one capture, no late compiles."""
    if not _MULTI:
        pytest.skip("spec decode is the TP path; run with MESH_DEVICE=P150x4")
    if not (os.environ.get("HF_MODEL") or os.environ.get("MODEL_WEIGHTS_DIR")):
        pytest.skip("set HF_MODEL (checkpoint dir)")
    from transformers import AutoTokenizer

    if _N_LAYERS:  # initialize_vllm_model has no n_layers arg; truncate through create_tt_model
        monkeypatch.setattr(
            qwen36_vllm, "create_tt_model", functools.partial(qwen36_vllm.create_tt_model, n_layers=_N_LAYERS)
        )
    mesh_device.enable_program_cache()
    hf_config = SimpleNamespace(_name_or_path=os.environ.get("HF_MODEL"))
    ref = Qwen36ForCausalLM.initialize_vllm_model(
        hf_config, mesh_device, max_batch_size=1, max_seq_len=MAX_MODEL_LEN, tt_data_parallel=1, optimizations=None
    )
    tokenizer = AutoTokenizer.from_pretrained(ref.model[0].args.CKPT_DIR, trust_remote_code=True)
    pool = _prompt_pool(tokenizer, 4096 + 512)
    p1 = _get_prompt(130, tokenizer)[0].tolist()
    p2 = pool[200 : 200 + 4300]  # two recorded 2048 chunks + a 204-token tail
    p3 = pool[:4096]  # two chunks, no tail
    assert len(p2) == 4300 and len(p3) == 4096
    reqs = [("P1", p1, 64), ("P2", p2, 48), ("P3", p3, 48)]

    # --- references first: Qwen36ForCausalLM on its own blocks, released before the MTP class starts ------ #
    refs = _ordinary_greedy(ref, reqs)
    del ref
    gc.collect()

    # --- startup: initialize_vllm_model (loads weights via HF_MODEL) ------------------------------ #
    cls = Qwen36MTPContractForCausalLM.initialize_vllm_model(
        hf_config, mesh_device, max_batch_size=1, max_seq_len=MAX_MODEL_LEN, tt_data_parallel=1, optimizations=None
    )
    model = cls.model[0]

    # --- allocate the KV cache the plugin way, then the 4 warmup calls in plugin order (or the lazy path) -- #
    probe = PluginLayout.for_model(model, num_blocks=1)
    width = math.ceil(MAX_MODEL_LEN / probe.block_size)
    layout = PluginLayout.for_model(model, num_blocks=width + probe.num_groups * (K + 1) + 8)
    kv = cls.allocate_kv_cache_per_layer(layout.per_layer_specs)
    if warmup == "lazy":  # enable_model_warmup=False: what the first prefill_forward runs
        cls._ensure_ready(kv)
    else:
        cls.warmup_model_prefill(enable_trace=False, kv_cache=kv, can_sample_on_device=False)
        cls.warmup_model_decode(
            enable_trace=False, kv_cache=kv, max_batch_size=1, num_blocks=width, can_sample_on_device=False
        )
        cls.warmup_model_prefill(enable_trace=True, kv_cache=kv, can_sample_on_device=False)
        cls.warmup_model_decode(
            enable_trace=True, kv_cache=kv, max_batch_size=1, num_blocks=width, can_sample_on_device=False
        )

    n_prog = mesh_device.num_program_cache_entries()
    assert model._vfy_capture_count == 1
    chunk_trace = model._chunked_trace_id
    assert chunk_trace is not None, "prefill chunk trace was not recorded"

    rng = torch.Generator().manual_seed(1234)

    def random_policy(step):
        return int(torch.randint(0, K + 1, (1,), generator=rng))

    policies = {"P1": lambda s: K, "P2": random_policy, "P3": lambda s: K}
    outs = {}
    mesh_device.set_program_cache_misses_allowed(False)
    try:
        for i, (name, prompt, max_new) in enumerate(reqs):
            out, st = PluginRunner(cls, kv, layout, seed=1 + i).run(prompt, max_new, policies[name])
            _log(name, out, st)
            assert len(out) == max_new, f"{name}: got {len(out)} tokens"
            _assert_matches_ref(name, out, *refs[name])
            outs[name] = out
            if name == "P2":
                assert any(0 < s["n"] < K for s in st["steps"]), "P2: no step with 0 < n < K"

        # P1 again on a different block pool: identical to the first P1 run
        out, st = PluginRunner(cls, kv, layout, seed=99).run(p1, 64, policies["P1"])
        _log("P1_repeat", out, st)
        assert out == outs["P1"], "P1 repeat on a different block pool differs from the first run"
        _assert_matches_ref("P1_repeat", out, *refs["P1"])

        # SAMPLED request (drafts withheld): all-narrow, sampled, drafter skipped, seed-reproducible
        skipped0 = cls._engine().skipped_propose_calls
        sampled = []
        for run_i in range(2):
            out, st = PluginRunner(cls, kv, layout, seed=50 + run_i, speculable=False, sample_seed=4242).run(
                p1, 48, lambda s: K
            )
            _log(f"P1_sampled{run_i}", out, st)
            assert len(out) == 48, f"sampled: got {len(out)} tokens"
            assert not any(s["verify"] for s in st["steps"]), "sampled request ran a verify step"
            assert st["steps"][-1]["skipped"] > st["steps"][0]["skipped"], "drafter not skipped after withholding"
            sampled.append(out)
        assert cls._engine().skipped_propose_calls > skipped0
        assert sampled[0] != outs["P1"][:48], "sampled output equals the greedy output"
        assert sampled[0] == sampled[1], "sampled tokens not reproducible with the same seed"

        # GREEDY P1 again: the skip mark was reset by release/prefill, so it speculates again
        out, st = PluginRunner(cls, kv, layout, seed=123).run(p1, 64, policies["P1"])
        _log("P1_after_sampled", out, st)
        assert out == outs["P1"], "greedy P1 after sampled requests differs from the first run"
        assert any(s["verify"] for s in st["steps"]), "greedy P1 never verified after sampled requests"
        mean = sum(s["m"] + 1 for s in st["steps"]) / len(st["steps"])
        assert mean > 1.5, f"greedy P1 stopped speculating after sampled requests (mean committed {mean:.2f})"
    finally:
        mesh_device.set_program_cache_misses_allowed(True)

    assert mesh_device.num_program_cache_entries() == n_prog, "program cache grew"
    assert model._vfy_capture_count == 1, "verify trace was re-captured"
    assert model._chunked_trace_id == chunk_trace, "prefill chunk trace was re-recorded"

    cls.release_persistent_capture()
    assert getattr(model, "_vfy_trace_id", None) is None, "verify trace not released"


# --- host tests (CPU) ---


def _cached_ckpt(repo="Qwen/Qwen3.8-27B"):
    """Local snapshot dir of ``repo`` (or HF_MODEL, a hub id or a local dir), or None when it is not cached."""
    name = os.environ.get("HF_MODEL", repo)
    if os.path.isdir(name):
        return name
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(name, local_files_only=True)
    except Exception:
        return None


CLS = Qwen36MTPContractForCausalLM
MAX_LEN = 16384


@pytest.fixture(scope="module")
def hf_config():
    ckpt = _cached_ckpt()
    if ckpt is None:
        pytest.skip("Qwen3.8-27B config not available")
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(ckpt)


@pytest.fixture(autouse=True)
def _mesh(request, monkeypatch):
    if "mesh_device" not in request.fixturenames:  # host tests only; the device test uses the real env
        monkeypatch.setenv("MESH_DEVICE", "P150x4")


def _vcfg(hf_config, k=3, max_len=MAX_LEN):
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf_config, max_model_len=max_len),
        speculative_config=SimpleNamespace(
            method="custom_class", model="vllm_tt_plugin.model_owned_drafter", num_speculative_tokens=k
        ),
    )


def test_spec_plan_accept(hf_config):
    """B=1, K=3 yields a SpecPlan whose byte costs match spec_memory_costs."""
    plan = CLS.spec_plan(_vcfg(hf_config), 1, 3)
    assert isinstance(plan, SpecPlan)
    assert plan.effective_k == 3
    assert "argmax_ids" in plan.accept_modes
    assert "logits" in plan.accept_modes
    assert plan.drafter_state == "internal"
    assert plan.supports_narrow_decode is True
    tc = hf_config.get_text_config()
    c = spec_memory_costs(tc, 4, 1, 3, max_seq_len=MAX_LEN, gdn_state_in_blocks=True)
    assert plan.extra_bytes_per_seq == int(c["per_seq"]["total"] + c["fixed"]["total"])
    assert c["per_seq"]["gdn_ring"] == 0 and c["per_seq"]["gdn_conv_windows"] == 0 and c["fixed"]["gdn_verify"] > 0
    assert plan.extra_bytes_per_token == int(c["per_token"]["total"])
    assert plan.extra_bytes_per_seq > 0 and plan.extra_bytes_per_token > 0


def test_spec_plan_larger_k_clamps(hf_config):
    """Requested K above the engine K is clamped to 3."""
    assert CLS.spec_plan(_vcfg(hf_config, 7), 1, 7).effective_k == 3


def test_spec_plan_small_k_rejected(hf_config):
    """Requested K below the engine K is rejected and reports K=3 as supported."""
    r = CLS.spec_plan(_vcfg(hf_config, 2), 1, 2)
    assert isinstance(r, SpecReject) and 3 in r.supported_k


def test_spec_plan_multi_seq(hf_config):
    """K=3 fits the 32-row verify tile up to 8 sequences; 9 is rejected with the supported K list."""
    plan = CLS.spec_plan(_vcfg(hf_config), 8, 3)
    assert isinstance(plan, SpecPlan) and plan.effective_k == 3
    c = spec_memory_costs(hf_config.get_text_config(), 4, 8, 3, max_seq_len=MAX_LEN, gdn_state_in_blocks=True)
    assert plan.extra_bytes_per_seq == math.ceil((c["per_seq"]["total"] + c["fixed"]["total"]) / 8)
    r = CLS.spec_plan(_vcfg(hf_config), 9, 3)
    assert isinstance(r, SpecReject) and "32-row" in r.reason and 3 not in r.supported_k


def test_spec_plan_no_mtp_rejected(hf_config):
    """A checkpoint without MTP layers is rejected."""
    tc = hf_config.get_text_config()
    cfg = _vcfg(SimpleNamespace(get_text_config=lambda: SimpleNamespace(**{**vars(tc), "mtp_num_hidden_layers": 0})))
    r = CLS.spec_plan(cfg, 1, 3)
    assert isinstance(r, SpecReject) and r.supported_k == () and "MTP" in r.reason


def test_spec_plan_broken_config_rejected():
    """A config missing text fields yields SpecReject, not an exception."""
    cfg = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(), max_model_len=MAX_LEN),
        speculative_config=None,
    )
    assert isinstance(CLS.spec_plan(cfg, 1, 3), SpecReject)
    assert isinstance(CLS.spec_plan(SimpleNamespace(), 1, 3), SpecReject)


@pytest.mark.parametrize("bs", [64, 128, 1024])
def test_spec_plan_any_block_size(hf_config, bs):
    """The block size comes from get_kv_cache_spec (one GDN state per block); --block-size is only its lower bound."""
    cfg = _vcfg(hf_config)
    cfg.cache_config = SimpleNamespace(block_size=bs)
    assert isinstance(CLS.spec_plan(cfg, 1, 3), SpecPlan)


def test_spec_plan_rejects_rows_past_the_spec_step_grid(hf_config, monkeypatch):
    """gdn_spec_step runs one core per (row, value head): max_num_seqs x Nv per device must fit its cores."""
    monkeypatch.setattr(qwen36_vllm, "_SPEC_STEP_CORES", 12 * 4 - 1)  # Nv = 48 / 4 = 12 at TP=4
    assert isinstance(CLS.spec_plan(_vcfg(hf_config), 3, 3), SpecPlan)
    r = CLS.spec_plan(_vcfg(hf_config), 4, 3)
    assert isinstance(r, SpecReject) and "gdn_spec_step" in r.reason


def test_kv_cache_spec_declares_k_speculative_blocks(hf_config, expect_error):
    """The MambaSpec groups take K = QWEN36_SPEC_K speculative blocks (Qwen36ForCausalLM: 0); a launch speculating
    another count fails at startup."""
    from vllm.v1.kv_cache_interface import MambaSpec

    def config(k):
        cfg = _vcfg(hf_config, k=k)
        cfg.cache_config = SimpleNamespace(block_size=64, mamba_block_size=None, mamba_cache_mode="align")
        return cfg

    def speculative_blocks(cls, cfg):
        return {s.num_speculative_blocks for s in cls.get_kv_cache_spec(cfg).values() if isinstance(s, MambaSpec)}

    assert speculative_blocks(CLS, config(CLS._SPEC_K)) == {CLS._SPEC_K}
    assert speculative_blocks(Qwen36ForCausalLM, config(CLS._SPEC_K)) == {0}
    with expect_error(ValueError, "num_speculative_tokens"):
        CLS.get_kv_cache_spec(config(CLS._SPEC_K + 4))


def test_decode_input_update_contract():
    """The spec class opts out of the reload commands; the base class keeps them."""
    assert CLS.decode_input_update_contract == 0
    assert Qwen36ForCausalLM.decode_input_update_contract == 1


def test_processor_info_text_only():
    """The spec class registers a text-only processor info; the base class keeps its own."""
    assert CLS._processor_factory.info is TT_Qwen3_5SpecProcessingInfo
    assert Qwen36ForCausalLM._processor_factory.info is TT_Qwen3_5ProcessingInfo


def test_plugin_admission_accepts_class(hf_config):
    """The plugin's resolve_speculative_plan admits the class with effective_k 3."""
    plan = resolve_speculative_plan(_vcfg(hf_config), CLS, CLS.model_capabilities, 1)
    assert isinstance(plan, SpecPlan) and plan.effective_k == 3


def test_capabilities():
    """Spec capabilities are declared on top of Qwen36ForCausalLM's: prefix caching and chunked prefill too, since the
    MTP class keeps its GDN state in the same vLLM-owned blocks."""
    caps = CLS.model_capabilities
    assert caps["supports_spec_decode"] is True
    assert "device_propose" in caps["spec_requirements"]
    assert tuple(caps["spec_hidden_handoff"]) == ("on_device",)
    assert caps["supports_async_decode"] is False
    assert caps["output_tokens_per_step"] == 1
    assert caps["supports_prefix_caching"] is True and caps["supports_chunked_prefill"] is True
    assert issubclass(CLS, Qwen36ForCausalLM)
    for key in Qwen36ForCausalLM.model_capabilities:
        assert key in caps


# ----------------------------------------------------------------------------- wrapper plumbing
class FakeEngine:
    def __init__(self, K=3, V=32):
        self.K, self.V = K, V
        self.calls = []

    def decode_forward(self, tokens, start_pos, slot_remap=None, **kw):
        kw["slot_remap"] = slot_remap
        self.calls.append(("decode_forward", tokens, start_pos, kw))
        logits = torch.arange(1 + self.K, dtype=torch.bfloat16)[None, :, None].expand(1, 1 + self.K, self.V)
        return VerifyResult(
            spec_mode=kw["spec_mode"], argmax_ids=torch.arange(1 + self.K, dtype=torch.int32)[None], logits=logits
        )

    def propose_draft_tokens(self, num_drafts, committed, positions, counts, hidden=None):
        self.calls.append(("propose", num_drafts, committed, positions, counts, hidden))
        return DraftResult(
            draft_token_ids=torch.tensor([[5, 6, 7]], dtype=torch.int32), num_valid=torch.tensor([3], dtype=torch.int32)
        )

    def prefill_hook(self, slot, row_tokens, page_table_row, final):
        self.calls.append(("prefill_hook", slot, row_tokens, page_table_row, final))
        return lambda hidden, chunk_start, valid_len: self.calls.append(("chunk", slot, chunk_start, valid_len))

    def finish_prefill(self, slot, end, final):
        self.calls.append(("finish_prefill", slot, end, final))

    def release(self, slot):
        self.calls.append(("release", slot))


BLOCK = 1024  # the attention / state block size get_kv_cache_spec picked
MAMBA_STATE = SimpleNamespace(groups=(), intermediate_prefill_mask=None)  # stand-in, forwarded as is


@pytest.fixture
def gen_eng(monkeypatch):
    """The class over a fake engine and a stand-in model whose prefill records its calls (Qwen36ForCausalLM's
    prefill_forward runs for real; the GDN state blocks are stand-ins)."""
    from models.demos.blackhole.qwen36.tt import gdn_block_state

    eng = FakeEngine()
    vocab = eng.V

    def prefill(tokens, pt, actual_len, vision_tokens=None, start_pos=0, on_chunk=None):
        eng.calls.append(("model_prefill", tokens.clone(), start_pos, actual_len))
        if on_chunk is not None:
            on_chunk(None, start_pos, actual_len - start_pos)
        return torch.full((1, 1, vocab), float(actual_len))

    blocks = SimpleNamespace(
        layout=SimpleNamespace(block_size=BLOCK),
        restore=lambda state, b: eng.calls.append(("restore", tuple(b))),
        write_state=lambda state, b: eng.calls.append(("write", tuple(b))),
    )
    model = SimpleNamespace(
        args=SimpleNamespace(max_seq_len=4096, max_batch_size=4, vocab_size=vocab),
        mesh_device=None,
        gdn_blocks=blocks,
        prefill_traced_chunked=prefill,
        prefill_chunk_size=2048,
    )
    gen = object.__new__(CLS)
    gen.model = [model]
    gen.data_parallel = 1
    monkeypatch.setattr(gen, "_engine", lambda: eng, raising=False)
    monkeypatch.setattr(gen, "_ensure_ready", lambda kv_cache=None: eng, raising=False)
    monkeypatch.setattr(ttnn, "to_torch", lambda t, **kw: t)
    monkeypatch.setattr(ttnn, "ConcatMeshToTensor", lambda *a, **kw: None)
    monkeypatch.setattr(ttnn, "deallocate", lambda t: None)
    monkeypatch.setattr(
        gdn_block_state,
        "row_blocks",
        lambda state, row: gdn_block_state.RowBlocks(start=(0,), dest=(10 + row,), checkpoint=None, checkpoint_pos=0),
    )
    return gen, eng


def _last(eng, name):
    return [c for c in eng.calls if c[0] == name][-1]


def test_decode_forward_verify(gen_eng):
    """decode_forward returns VerifyOutput(argmax_ids) and passes widened page table and state to the engine."""
    gen, eng = gen_eng
    tokens = torch.tensor([[1, 2, 3, 4]], dtype=torch.int32)
    start_pos = torch.tensor([10], dtype=torch.int32)
    pt = torch.arange(1, 5, dtype=torch.int32).reshape(1, 4)
    nv, ac = torch.tensor([3], dtype=torch.int32), torch.tensor([2], dtype=torch.int32)
    out = gen.decode_forward(
        tokens=tokens, start_pos=start_pos, page_table=pt, kv_cache=None, enable_trace=True, read_from_device=False,
        num_valid_drafts=nv, accepted_counts=ac, spec_mode="argmax_ids", reload_inputs=True,
        reload_page_table=False, reload_sampling_params=False, reset_sampling_state=False, sampling_params=None,
        mamba_state=MAMBA_STATE,
    )  # fmt: skip
    assert isinstance(out, VerifyOutput)
    assert out.spec_mode == "argmax_ids" and out.hidden is None
    assert out.argmax_ids.tolist() == [[0, 1, 2, 3]]
    _, t, sp, kw = _last(eng, "decode_forward")
    assert torch.equal(t, tokens) and torch.equal(sp, start_pos)
    assert torch.equal(kw["num_valid_drafts"], nv) and torch.equal(kw["accepted_counts"], ac)
    assert kw["mamba_state"] is MAMBA_STATE
    w = spec_table_width(4096, 3, BLOCK)
    table = kw["page_table"]
    assert table.shape == (1, w) and table.dtype == torch.int32
    assert torch.equal(table[0, :4], pt[0]) and not table[0, 4:].any()


def test_decode_forward_logits_mode_carries_logits(gen_eng):
    """spec_mode="logits" puts the engine logits into VerifyOutput; argmax_ids mode leaves them None."""
    gen, eng = gen_eng
    kw = dict(
        tokens=torch.tensor([[1, 2, 3, 4]], dtype=torch.int32), start_pos=torch.tensor([10], dtype=torch.int32),
        page_table=torch.arange(1, 9, dtype=torch.int32).reshape(1, 8),
        num_valid_drafts=torch.tensor([3], dtype=torch.int32), accepted_counts=torch.tensor([1], dtype=torch.int32),
        mamba_state=MAMBA_STATE,
    )  # fmt: skip
    out = gen.decode_forward(**kw, spec_mode="logits")
    assert tuple(out.logits.shape) == (1, 4, eng.V)
    assert gen.decode_forward(**kw, spec_mode="argmax_ids").logits is None


def test_decode_forward_slot_remap_verify(gen_eng):
    """A verify step passes slot_remap into a single engine.decode_forward call; None stays None."""
    gen, eng = gen_eng
    kw = dict(
        tokens=torch.tensor([[1, 2, 3, 4]], dtype=torch.int32), start_pos=torch.tensor([10], dtype=torch.int32),
        page_table=torch.arange(1, 9, dtype=torch.int32).reshape(1, 8),
        num_valid_drafts=torch.tensor([3], dtype=torch.int32), accepted_counts=torch.tensor([1], dtype=torch.int32),
        spec_mode="argmax_ids", mamba_state=MAMBA_STATE,
    )  # fmt: skip
    gen.decode_forward(**kw)
    assert [c[0] for c in eng.calls] == ["decode_forward"] and eng.calls[0][3]["slot_remap"] is None
    eng.calls.clear()
    gen.decode_forward(**kw, slot_remap=torch.tensor([1, 0]))
    assert [c[0] for c in eng.calls] == ["decode_forward"]
    assert eng.calls[0][3]["slot_remap"].tolist() == [1, 0]


def test_decode_forward_slot_remap_narrow(gen_eng):
    """A narrow step passes slot_remap into a single engine.decode_forward call."""
    gen, eng = gen_eng
    gen.decode_forward(
        tokens=torch.tensor([7]), start_pos=torch.tensor([10]), page_table=torch.ones(1, 8, dtype=torch.int32),
        slot_remap=torch.tensor([1, 0]), mamba_state=MAMBA_STATE,
    )  # fmt: skip
    assert [c[0] for c in eng.calls] == ["decode_forward"]
    assert eng.calls[0][3]["slot_remap"].tolist() == [1, 0]


@pytest.mark.parametrize("tok_shape", [(1,), (1, 1)])
def test_decode_forward_narrow(gen_eng, tok_shape):
    """No spec_mode: a 1-column verify in logits mode; returns host float logits [B, 1, V]."""
    gen, eng = gen_eng
    pt = torch.arange(1, 9, dtype=torch.int32).reshape(1, 8)
    out = gen.decode_forward(
        tokens=torch.full(tok_shape, 7, dtype=torch.int32), start_pos=torch.tensor([10], dtype=torch.int32),
        page_table=pt, enable_trace=True, read_from_device=True, sampling_params=None,
        num_valid_drafts=torch.zeros(1, dtype=torch.int32), accepted_counts=torch.ones(1, dtype=torch.int32),
        mamba_state=MAMBA_STATE,
    )  # fmt: skip
    assert isinstance(out, torch.Tensor) and out.dtype == torch.float32 and tuple(out.shape) == (1, 1, eng.V)
    _, t, sp, kw = _last(eng, "decode_forward")
    assert t.tolist() == [[7, -1, -1, -1]] and sp.tolist() == [[10, -1, -1, -1]]
    assert kw["spec_mode"] == "logits" and kw["num_valid_drafts"].tolist() == [0]
    assert kw["accepted_counts"].tolist() == [1] and kw["mamba_state"] is MAMBA_STATE


def test_decode_forward_needs_mamba_state(gen_eng, expect_error):
    """The GDN state is in vLLM's blocks: a decode without mamba_state is refused before the engine runs."""
    gen, eng = gen_eng
    with expect_error(ValueError, "needs mamba_state"):
        gen.decode_forward(tokens=torch.tensor([7]), start_pos=torch.tensor([10]), page_table=torch.ones(1, 8))
    assert eng.calls == []


def test_decode_forward_narrow_counts(gen_eng, expect_error):
    """Counts are optional on a narrow step (default 1) and are passed through; drafts are rejected."""
    gen, eng = gen_eng
    base = dict(
        tokens=torch.tensor([7]),
        start_pos=torch.tensor([10]),
        page_table=torch.ones(1, 8, dtype=torch.int32),
        mamba_state=MAMBA_STATE,
    )
    assert gen.decode_forward(**base).shape[:2] == (1, 1)
    assert _last(eng, "decode_forward")[3]["accepted_counts"].tolist() == [1]
    gen.decode_forward(**base, accepted_counts=torch.tensor([2], dtype=torch.int32))
    kw = _last(eng, "decode_forward")[3]
    assert kw["accepted_counts"].tolist() == [2] and kw["accepted_counts"].dtype == torch.int32
    assert kw["num_valid_drafts"].tolist() == [0]
    with expect_error(ValueError, ".*"):
        gen.decode_forward(**base, num_valid_drafts=torch.tensor([1], dtype=torch.int32))


@pytest.mark.parametrize("bad", ["nv_int64", "ac_zero"])
def test_decode_forward_verify_rejects_bad_side_tensors(gen_eng, bad, expect_error):
    """Invalid side tensors raise ValueError before the engine sees any call."""
    gen, eng = gen_eng
    nv, ac = torch.tensor([3], dtype=torch.int32), torch.tensor([1], dtype=torch.int32)
    if bad == "nv_int64":
        nv, msg = nv.long(), "num_valid_drafts must be int32"
    else:
        ac, msg = torch.tensor([0], dtype=torch.int32), "accepted_counts entries must lie in"
    with expect_error(ValueError, msg):
        gen.decode_forward(
            tokens=torch.tensor([[1, 2, 3, 4]], dtype=torch.int32), start_pos=torch.tensor([10], dtype=torch.int32),
            page_table=torch.arange(1, 9, dtype=torch.int32).reshape(1, 8),
            num_valid_drafts=nv, accepted_counts=ac, spec_mode="argmax_ids", mamba_state=MAMBA_STATE,
        )  # fmt: skip
    assert eng.calls == []


def test_decode_forward_narrow_rejects_out_of_range_counts(gen_eng, expect_error):
    """Narrow accepted_counts outside [1, 1+K] raises ValueError before the engine call."""
    gen, eng = gen_eng
    base = dict(
        tokens=torch.tensor([7]),
        start_pos=torch.tensor([10]),
        page_table=torch.ones(1, 8, dtype=torch.int32),
        mamba_state=MAMBA_STATE,
    )
    for bad in (0, 5):
        with expect_error(ValueError, "accepted_counts entries must lie in"):
            gen.decode_forward(**base, accepted_counts=torch.tensor([bad], dtype=torch.int32))
    assert eng.calls == []


def test_warmup_model_decode_enable_trace_routing(monkeypatch):
    """enable_trace (kwarg or 2nd positional) captures; otherwise prepares."""
    gen = object.__new__(CLS)
    gen.data_parallel = 1
    eng = SimpleNamespace(captured=False, prepared=False, log=[])
    eng.capture_decode = lambda: eng.log.append("capture")
    eng.prepare_decode = lambda: eng.log.append("prepare")
    monkeypatch.setattr(gen, "_engine", lambda: eng, raising=False)
    gen.warmup_model_decode(None, False)
    gen.warmup_model_decode(kv_cache=None, enable_trace=True)
    gen.warmup_model_decode(None, True, can_sample_on_device=False)
    assert eng.log == ["prepare", "capture", "capture"]


def test_ensure_ready_prepares_before_recording(monkeypatch):
    """Lazy warmup follows the plugin order: prefill and decode prepared before any trace is recorded."""
    gen = object.__new__(CLS)
    gen.data_parallel = 1
    eng = SimpleNamespace(prefill_prepared=False, prepared=False, prefill_recorded=False, captured=False, log=[])

    def step(flag, name):
        def run(*a, **k):
            eng.log.append(name)
            setattr(eng, flag, True)

        return run

    eng.prepare_decode = step("prepared", "prepare_decode")
    eng.capture_decode = step("captured", "capture_decode")
    monkeypatch.setattr(gen, "_engine", lambda: eng, raising=False)

    def warmup_prefill(kv_cache, enable_trace):
        if enable_trace:
            step("prefill_recorded", "record_prefill")()
        else:
            step("prefill_prepared", "prepare_prefill")()

    monkeypatch.setattr(gen, "warmup_model_prefill", warmup_prefill, raising=False)
    assert gen._ensure_ready("kv") is eng
    assert eng.log == ["prepare_prefill", "prepare_decode", "record_prefill", "capture_decode"]
    gen._ensure_ready("kv")
    assert len(eng.log) == 4


def test_propose_draft_tokens(gen_eng):
    """propose_draft_tokens wraps the engine result in a DraftOutput."""
    gen, eng = gen_eng
    committed = torch.tensor([[9, 0, 0, 0]], dtype=torch.int32)
    pos = torch.tensor([[11, 0, 0, 0]], dtype=torch.int32)
    counts = torch.tensor([1], dtype=torch.int32)
    out = gen.propose_draft_tokens(3, committed, pos, counts, hidden=None)
    assert isinstance(out, DraftOutput)
    assert out.draft_token_ids.dtype == torch.int32 and tuple(out.draft_token_ids.shape) == (1, 3)
    assert out.num_valid.tolist() == [3]
    c = _last(eng, "propose")
    assert c[1] == 3 and c[5] is None and torch.equal(c[2], committed)


def _prefill(gen, n, ends, mask=None, slots=None, **kw):
    tokens = torch.arange(20 * n, dtype=torch.long).reshape(n, 20)
    pt = torch.arange(1, 1 + 5 * n, dtype=torch.int32).reshape(n, 5)
    state = SimpleNamespace(groups=(), intermediate_prefill_mask=None if mask is None else torch.tensor(mask))
    return gen.prefill_forward(
        tokens, pt, None, prompt_lens=np.array(ends), empty_slots=slots, start_pos=np.array([0] * n),
        mamba_state=state, **kw,
    )  # fmt: skip


def test_prefill_forward(gen_eng):
    """Qwen36ForCausalLM's prefill runs with the engine's chunk hook; the row's end state goes to its block, and the
    engine learns that the row ended its prompt."""
    gen, eng = gen_eng
    logits, rope = _prefill(gen, 1, [7], slots=[0])
    assert logits.dtype == torch.float32 and tuple(logits.shape) == (1, 1, eng.V) and rope is not None
    names = [c[0] for c in eng.calls]
    assert names == ["prefill_hook", "model_prefill", "chunk", "write", "finish_prefill"]
    _, slot, toks, table, final = _last(eng, "prefill_hook")
    assert slot == 0 and toks.tolist() == list(range(7)) and final is True
    assert table.shape == (1, spec_table_width(4096, 3, BLOCK))
    assert _last(eng, "finish_prefill") == ("finish_prefill", 0, 7, True)


def test_prefill_forward_rejects_visual(gen_eng, expect_error):
    """Image/video inputs raise ValueError before the engine is touched."""
    gen, eng = gen_eng
    for key in ("pixel_values", "pixel_values_videos"):
        with expect_error(ValueError, "text-only"):
            _prefill(gen, 1, [7], slots=[0], **{key: [torch.zeros(1)]})
    assert eng.calls == []


def test_prefill_forward_multi_row(gen_eng):
    """N rows prefill into empty_slots in order with their own tokens and table rows; a chunk that does not end its
    prompt (intermediate_prefill_mask True) keeps no last row and does not start the drafter."""
    gen, eng = gen_eng
    logits, rope = _prefill(gen, 2, [7, 12], mask=[True, False], slots=[3, 1])
    assert tuple(logits.shape) == (2, 1, eng.V) and rope.shape == (2,)
    hooks = [c for c in eng.calls if c[0] == "prefill_hook"]
    assert [(c[1], c[4]) for c in hooks] == [(3, False), (1, True)]
    assert hooks[0][2].tolist() == list(range(7)) and hooks[1][2].tolist() == list(range(20, 32))
    for i, c in enumerate(hooks):
        assert c[3].shape == (1, spec_table_width(4096, 3, BLOCK))
        assert torch.equal(c[3][0, :5], torch.arange(1 + 5 * i, 6 + 5 * i, dtype=torch.int32))
    assert [c[1:] for c in eng.calls if c[0] == "finish_prefill"] == [(3, 7, False), (1, 12, True)]


def test_release_and_slot_moves(gen_eng):
    """release_request forwards to engine.release; note_state_slots_moved needs no engine work."""
    gen, eng = gen_eng
    gen.release_request(0)
    assert _last(eng, "release") == ("release", 0)
    n = len(eng.calls)
    assert gen.note_state_slots_moved({0: 0}) is None
    assert gen.note_state_slots_moved({0: 1, 1: 0}) is None
    assert len(eng.calls) == n
