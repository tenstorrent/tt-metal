# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Qwen36MTPContractForCausalLM under the vLLM TT plugin (skipped without vllm / vllm_tt_plugin).

Host tests (CPU, fake engine): spec_plan, capabilities, wrapper plumbing.
Device test: drive the class exactly as the plugin runner (#131) calls a model, without a server:
initialize_vllm_model -> allocate_kv_cache -> 2-phase warmup -> prefill_forward / decode_forward /
propose_draft_tokens / release_request per request -> release_persistent_capture. Output must stay lossless
vs model-level plain greedy (exact up to the first bf16 near-tie flip).
Runs under TT_METAL_TRACE_ALLOC_TRACKING=1 unchanged.
"""

import functools
import math
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from loguru import logger

pytest.importorskip("vllm")
pytest.importorskip("vllm_tt_plugin")

from vllm_tt_plugin.spec_admission import resolve_speculative_plan  # noqa: E402
from vllm_tt_plugin.spec_decode import DraftOutput, SpecPlan, SpecReject, VerifyOutput  # noqa: E402

import ttnn  # noqa: E402
from models.common.utility_functions import run_for_blackhole  # noqa: E402
from models.demos.blackhole.qwen36.demo.text_demo import (  # noqa: E402
    _MESH_SHAPE,
    _MULTI,
    BLOCK_SIZE,
    DEVICE_PARAMS,
    _get_prompt,
)
from models.demos.blackhole.qwen36.tests.spec_helpers import (  # noqa: E402
    PAD,
    _assert_matches_ref,
    _perm_table,
    _plain_greedy,
    _prompt_pool,
)
from models.demos.blackhole.qwen36.tests.test_spec_lossless import _N_LAYERS  # noqa: E402
from models.demos.blackhole.qwen36.tt import qwen36_vllm  # noqa: E402
from models.demos.blackhole.qwen36.tt.qwen36_vllm import Qwen36ForCausalLM, Qwen36MTPContractForCausalLM  # noqa: E402
from models.demos.blackhole.qwen36.tt.spec_engine import (  # noqa: E402
    DraftResult,
    VerifyResult,
    spec_memory_costs,
    spec_table_width,
)

K = 3
MAX_MODEL_LEN = 8192
W = math.ceil(MAX_MODEL_LEN / BLOCK_SIZE)  # page-table width (128)
NUM_BLOCKS = W + 8  # pool size handed to allocate_kv_cache


def _sample(logits, gen, temperature=0.7, top_p=0.95):
    """Host temperature + top-p sample of one [V] logits row."""
    probs = torch.softmax(logits.float() / temperature, dim=-1)
    sp, si = probs.sort(descending=True)
    keep = (sp.cumsum(-1) - sp) < top_p  # always keeps the top token
    sp = sp * keep
    return int(si[torch.multinomial(sp / sp.sum(), 1, generator=gen)])


class PluginRunner:
    """Host-side B=1 mirror of the plugin runner's calls into the model class.

    A step is a verify iff it has drafts or the previous commit was > 1 token; otherwise a narrow decode
    (no spec_mode, host-sampled from the returned logits) followed by propose_draft_tokens. A request with
    speculable=False never has its drafts published, so it is always narrow."""

    def __init__(self, model, kv, seed, speculable=True, sample_seed=0):
        self.m, self.kv, self.K = model, kv, K
        self.speculable = speculable
        self.gen = torch.Generator().manual_seed(sample_seed)
        # Seeded shuffled pool of real blocks; block 0 is vLLM's null block and is never handed out.
        g = torch.Generator().manual_seed(seed)
        self.free = (torch.randperm(NUM_BLOCKS - 1, generator=g) + 1).tolist()
        self.blocks = []

    def _table(self, last_pos):
        """[1, W] int32 table covering positions up to last_pos + (1+K) + (K+1) lookahead, zero padded."""
        need = min(W, math.ceil((last_pos + (1 + K) + (K + 1)) / BLOCK_SIZE))
        while len(self.blocks) < need:
            self.blocks.append(self.free.pop())
        return torch.tensor([self.blocks + [0] * (W - len(self.blocks))], dtype=torch.int32)

    def _pick(self, logits_row):
        return int(logits_row.argmax()) if self.speculable else _sample(logits_row, self.gen)

    def run(self, prompt, max_new, n_policy):
        m, cap = self.m, W * BLOCK_SIZE
        P = len(prompt)
        tokens = torch.tensor([prompt], dtype=torch.int32)
        logits, _ = m.prefill_forward(
            tokens,
            self._table(P - 1),
            self.kv,
            prompt_lens=torch.tensor([P]).numpy(),
            start_pos=torch.tensor([0]).numpy(),
            empty_slots=[0],
            enable_trace=True,
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
def test_spec_vllm_class_runner(mesh_device, monkeypatch):
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

    # --- startup: initialize_vllm_model (loads weights via HF_MODEL) ------------------------------ #
    hf_config = SimpleNamespace(_name_or_path=os.environ.get("HF_MODEL"))
    cls = Qwen36MTPContractForCausalLM.initialize_vllm_model(
        hf_config, mesh_device, max_batch_size=1, max_seq_len=MAX_MODEL_LEN, tt_data_parallel=1, optimizations=None
    )
    model = cls.model[0]
    tokenizer = AutoTokenizer.from_pretrained(model.args.CKPT_DIR, trust_remote_code=True)

    # Plugin shape: (num_blocks, num_kv_heads // min(num_devices, num_kv_heads), block_size, head_size).
    tc = model.args.hf_config.get_text_config()
    n_kv = tc.num_key_value_heads
    kv_shape = (NUM_BLOCKS, n_kv // min(model.num_devices, n_kv), BLOCK_SIZE, model.args.head_dim)
    assert kv_shape[1] == model.args.n_local_kv_heads, f"plugin kv heads {kv_shape[1]} != model local heads"

    pool = _prompt_pool(tokenizer, 4096 + 512)
    p1 = _get_prompt(130, tokenizer)[0].tolist()
    p2 = pool[200 : 200 + 4300]  # two recorded 2048 chunks + a 204-token tail
    p3 = pool[:4096]  # two chunks, no tail
    assert len(p2) == 4300 and len(p3) == 4096
    reqs = [("P1", p1, 64), ("P2", p2, 48), ("P3", p3, 48)]

    # --- plain references first (model-level, own KV alloc/free), before the class allocates KV ----- #
    refs = {
        name: _plain_greedy(model, list(kv_shape), prompt, _perm_table(W, seed=7 + i), max_new)
        for i, (name, prompt, max_new) in enumerate(reqs)
    }

    # --- allocate the KV cache the plugin way, then the 4 warmup calls in plugin order -------------- #
    kv = cls.allocate_kv_cache(kv_shape, ttnn.bfloat16, len(model._attention_layer_indices))
    cls.warmup_model_prefill(enable_trace=False, kv_cache=kv, can_sample_on_device=False)
    cls.warmup_model_decode(enable_trace=False, kv_cache=kv, max_batch_size=1, num_blocks=W, can_sample_on_device=False)
    cls.warmup_model_prefill(enable_trace=True, kv_cache=kv, can_sample_on_device=False)
    cls.warmup_model_decode(enable_trace=True, kv_cache=kv, max_batch_size=1, num_blocks=W, can_sample_on_device=False)

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
            out, st = PluginRunner(cls, kv, seed=1 + i).run(prompt, max_new, policies[name])
            _log(name, out, st)
            assert len(out) == max_new, f"{name}: got {len(out)} tokens"
            _assert_matches_ref(name, out, *refs[name])
            outs[name] = out
            if name == "P2":
                assert any(0 < s["n"] < K for s in st["steps"]), "P2: no step with 0 < n < K"

        # P1 again on a different block pool: identical to the first P1 run
        out, st = PluginRunner(cls, kv, seed=99).run(p1, 64, policies["P1"])
        _log("P1_repeat", out, st)
        assert out == outs["P1"], "P1 repeat on a different block pool differs from the first run"
        _assert_matches_ref("P1_repeat", out, *refs["P1"])

        # SAMPLED request (drafts withheld): all-narrow, sampled, drafter skipped, seed-reproducible
        skipped0 = cls._engine().skipped_propose_calls
        sampled = []
        for run_i in range(2):
            out, st = PluginRunner(cls, kv, seed=50 + run_i, speculable=False, sample_seed=4242).run(
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
        out, st = PluginRunner(cls, kv, seed=123).run(p1, 64, policies["P1"])
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

CKPT = (
    "/local/ttuser/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
)
CLS = Qwen36MTPContractForCausalLM
MAX_LEN = 16384


@pytest.fixture(scope="module")
def hf_config():
    if not os.path.isdir(CKPT):
        pytest.skip("Qwen3.8-27B config not available")
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(CKPT)


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
    assert plan.drafter_state == "internal"
    assert plan.supports_narrow_decode is True
    tc = hf_config.get_text_config()
    c = spec_memory_costs(tc, 4, 1, 3, max_seq_len=MAX_LEN)
    assert plan.extra_bytes_per_seq == int(c["per_seq"]["total"] + c["fixed"]["total"])
    assert plan.extra_bytes_per_token == int(c["per_token"]["total"])
    assert plan.extra_bytes_per_seq > 0 and plan.extra_bytes_per_token > 0


def test_spec_plan_larger_k_clamps(hf_config):
    """Requested K above the engine K is clamped to 3."""
    assert CLS.spec_plan(_vcfg(hf_config, 7), 1, 7).effective_k == 3


def test_spec_plan_small_k_rejected(hf_config):
    """Requested K below the engine K is rejected and reports K=3 as supported."""
    r = CLS.spec_plan(_vcfg(hf_config, 2), 1, 2)
    assert isinstance(r, SpecReject) and 3 in r.supported_k


def test_spec_plan_multi_seq_rejected(hf_config):
    """max_num_seqs != 1 is rejected."""
    assert isinstance(CLS.spec_plan(_vcfg(hf_config), 2, 3), SpecReject)


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


def test_plugin_admission_accepts_class(hf_config):
    """The plugin's resolve_speculative_plan admits the class with effective_k 3."""
    plan = resolve_speculative_plan(_vcfg(hf_config), CLS, CLS.model_capabilities, 1)
    assert isinstance(plan, SpecPlan) and plan.effective_k == 3


def test_capabilities():
    """Spec capabilities are declared and the base class capabilities are preserved."""
    caps = CLS.model_capabilities
    assert caps["supports_spec_decode"] is True
    assert "device_propose" in caps["spec_requirements"]
    assert tuple(caps["spec_hidden_handoff"]) == ("on_device",)
    assert caps["supports_async_decode"] is False
    assert caps["output_tokens_per_step"] == 1
    for key in Qwen36ForCausalLM.model_capabilities:
        assert key in caps


# ----------------------------------------------------------------------------- wrapper plumbing
class FakeEngine:
    def __init__(self, K=3, V=32):
        self.K, self.V = K, V
        self.calls = []

    def decode_forward(self, tokens, start_pos, **kw):
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

    def prefill(self, slot, tokens, page_table):
        self.calls.append(("prefill", slot, tokens, page_table))
        return torch.zeros(self.V, dtype=torch.bfloat16)

    def release(self, slot):
        self.calls.append(("release", slot))


@pytest.fixture
def gen_eng(monkeypatch):
    gen = object.__new__(CLS)
    gen.model = [SimpleNamespace(args=SimpleNamespace(max_seq_len=4096))]
    eng = FakeEngine()
    monkeypatch.setattr(gen, "_engine", lambda: eng, raising=False)
    monkeypatch.setattr(gen, "_ensure_ready", lambda kv_cache=None: eng, raising=False)
    return gen, eng


def _last(eng, name):
    return [c for c in eng.calls if c[0] == name][-1]


def test_decode_forward_verify(gen_eng):
    """decode_forward returns VerifyOutput(argmax_ids) and passes widened page table and state to the engine."""
    gen, eng = gen_eng
    tokens = torch.tensor([[1, 2, 3, 4]], dtype=torch.int32)
    start_pos = torch.tensor([10], dtype=torch.int32)
    pt = torch.arange(1, 9, dtype=torch.int32).reshape(1, 8)
    nv, ac = torch.tensor([3], dtype=torch.int32), torch.tensor([2], dtype=torch.int32)
    out = gen.decode_forward(
        tokens=tokens, start_pos=start_pos, page_table=pt, kv_cache=None, enable_trace=True, read_from_device=False,
        num_valid_drafts=nv, accepted_counts=ac, spec_mode="argmax_ids", reload_inputs=True,
        reload_page_table=False, reload_sampling_params=False, reset_sampling_state=False, sampling_params=None,
    )  # fmt: skip
    assert isinstance(out, VerifyOutput)
    assert out.spec_mode == "argmax_ids" and out.hidden is None
    assert out.argmax_ids.tolist() == [[0, 1, 2, 3]]
    _, t, sp, kw = _last(eng, "decode_forward")
    assert torch.equal(t, tokens) and torch.equal(sp, start_pos)
    assert torch.equal(kw["num_valid_drafts"], nv) and torch.equal(kw["accepted_counts"], ac)
    w = spec_table_width(4096, 3, 64)
    table = kw["page_table"]
    assert table.shape == (1, w) and table.dtype == torch.int32
    assert torch.equal(table[0, :8], pt[0]) and not table[0, 8:].any()


def test_decode_forward_logits_mode_carries_logits(gen_eng):
    """spec_mode="logits" puts the engine logits into VerifyOutput; argmax_ids mode leaves them None."""
    gen, eng = gen_eng
    kw = dict(
        tokens=torch.tensor([[1, 2, 3, 4]], dtype=torch.int32), start_pos=torch.tensor([10], dtype=torch.int32),
        page_table=torch.arange(1, 9, dtype=torch.int32).reshape(1, 8),
        num_valid_drafts=torch.tensor([3], dtype=torch.int32), accepted_counts=torch.tensor([1], dtype=torch.int32),
    )  # fmt: skip
    out = gen.decode_forward(**kw, spec_mode="logits")
    assert tuple(out.logits.shape) == (1, 4, eng.V)
    assert gen.decode_forward(**kw, spec_mode="argmax_ids").logits is None


@pytest.mark.parametrize("tok_shape", [(1,), (1, 1)])
def test_decode_forward_narrow(gen_eng, tok_shape):
    """No spec_mode: a 1-column verify in logits mode; returns host float logits [B, 1, V]."""
    gen, eng = gen_eng
    pt = torch.arange(1, 9, dtype=torch.int32).reshape(1, 8)
    out = gen.decode_forward(
        tokens=torch.full(tok_shape, 7, dtype=torch.int32), start_pos=torch.tensor([10], dtype=torch.int32),
        page_table=pt, enable_trace=True, read_from_device=True, sampling_params=None,
        num_valid_drafts=torch.zeros(1, dtype=torch.int32), accepted_counts=torch.ones(1, dtype=torch.int32),
    )  # fmt: skip
    assert isinstance(out, torch.Tensor) and out.dtype == torch.float32 and tuple(out.shape) == (1, 1, eng.V)
    _, t, sp, kw = _last(eng, "decode_forward")
    assert t.tolist() == [[7, -1, -1, -1]] and sp.tolist() == [[10, -1, -1, -1]]
    assert kw["spec_mode"] == "logits" and kw["num_valid_drafts"].tolist() == [0]
    assert kw["accepted_counts"].tolist() == [1]


def test_decode_forward_narrow_without_counts_and_bad_counts(gen_eng, expect_error):
    """Counts are optional on a narrow step; drafts or accepted_counts != 1 are rejected."""
    gen, _ = gen_eng
    base = dict(tokens=torch.tensor([7]), start_pos=torch.tensor([10]), page_table=torch.ones(1, 8, dtype=torch.int32))
    assert gen.decode_forward(**base).shape[:2] == (1, 1)
    with expect_error(ValueError, ".*"):
        gen.decode_forward(**base, num_valid_drafts=torch.tensor([1]))
    with expect_error(ValueError, ".*"):
        gen.decode_forward(**base, accepted_counts=torch.tensor([2]))


def test_warmup_model_decode_enable_trace_routing(monkeypatch):
    """enable_trace (kwarg or 2nd positional) captures; otherwise prepares."""
    gen = object.__new__(CLS)
    eng = SimpleNamespace(captured=False, prepared=False, log=[])
    eng.capture_decode = lambda: eng.log.append("capture")
    eng.prepare_decode = lambda: eng.log.append("prepare")
    monkeypatch.setattr(gen, "_engine", lambda: eng, raising=False)
    gen.warmup_model_decode(None, False)
    gen.warmup_model_decode(kv_cache=None, enable_trace=True)
    gen.warmup_model_decode(None, True, can_sample_on_device=False)
    assert eng.log == ["prepare", "capture", "capture"]


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


def test_prefill_forward(gen_eng):
    """prefill_forward returns float logits [1,1,V] plus rope deltas and routes slot/tokens to the engine."""
    gen, eng = gen_eng
    L = 7
    tokens = torch.arange(20, dtype=torch.long).reshape(1, 20)
    pt = torch.arange(1, 6, dtype=torch.int32).reshape(1, 5)
    logits, rope = gen.prefill_forward(
        tokens, pt, None, prompt_lens=np.array([L]), empty_slots=[0], start_pos=np.array([0])
    )
    assert logits.dtype == torch.float32 and tuple(logits.shape) == (1, 1, eng.V)
    assert rope is not None
    _, slot, toks, table = _last(eng, "prefill")
    assert slot == 0 and toks.tolist() == list(range(L))
    assert table.shape == (1, spec_table_width(4096, 3, 64))


def test_release_and_slot_moves(gen_eng, expect_error):
    """release_request forwards to engine.release; identity slot moves are no-ops, others raise."""
    gen, eng = gen_eng
    gen.release_request(0)
    assert _last(eng, "release") == ("release", 0)
    assert gen.note_state_slots_moved({0: 0}) is None
    with expect_error(NotImplementedError, ".*"):
        gen.note_state_slots_moved({0: 1})
