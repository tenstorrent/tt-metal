# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""vLLM (tt-vllm-plugin) adapter of Hy4-Preview on the 4x8 Blackhole galaxy, PRODUCT version (task product-integ): ``Hy4Engine`` (tt/hy4_serve_engine.py) = sequence-sharded SCALED_FP8 pool (1M tokens per
mesh row), DIRECT-TO-POOL chunked prefill (any prompt length up to the pool, MTP prompt KV written per chunk), device speculative rounds through the model's MTP head, device sampling.

Registered with the plugin as ``TTHYV4ForCausalLM`` through ``../vllm_metadata.json`` (``EXTRA_MODELS_DIR`` = ``models/demos/blackhole``). Model environment: HY4_POOL=1 HY4_POOL_PF=1 HY4_KV_FMT=scaled_fp8
HY4_OPT=all HY4_IHC_FUSED=1 HY4_FUSE=sparse,xchg HY4_XM_NSPLIT=8 HY4_SKINNY=shared,router,oproj,hproj HY4_PF_OPT=all,moeall MOE_RING_FAST=1 (tests/run_product_server.sh sets them).

MODES (decided at construction from the server's own command line / HY4_SPEC_K)
  * SPEC (``--speculative-config '{"method":"custom_class","model":"vllm_tt_plugin.model_owned_drafter","num_speculative_tokens":3}' --no-async-scheduling``): the plugin's model-owned drafter contract
    (docs/SPEC_DECODE_CONTRACT.md), the same shape as the DSV4.1 adapter: a step carries a verify block [B, 1+K] when some row has drafts, else an ordinary [B, 1] decode (narrow decode: served as a forced single-token round). The device runs one traced ROUND per verify (verify S = K+1 rows through
    the 78 layers, accept, MTP draft of the next K tokens, all state resident); the adapter returns the emitted ids a_0..a_K as ``argmax_ids`` (the plugin's greedy walk reaches the same accept count m as the
    device) and hands the drafts of the round out in ``propose_draft_tokens``. A request with temperature > 0 is served by the lossless SAMPLED round (draft-from-q rejection sampling, temperature only; the
    plugin capability ``supports_sampled_verify``: the returned ids have the shape of the greedy walk: a rejected draw never equals its draft). Penalties / logprobs / grammar are refused by the plugin
    (not speculable, no narrow decode).
  * PLAIN (no speculative config, or HY4_SPEC_K=0): ``Hy4PoolDecoder`` S=1 + the device decode loop (resident state, in-trace sampler, one trace per sampler variant); host sampling is NOT available (refused).
  Prefill (both): direct-to-pool, whole prompts per call (no chunked prefill / no prefix cache), the first token is sampled on the host from the last-prompt logits row.

LIMITS: prompt + generation <= max_model_len (<= 1,048,576), max_num_seqs <= 4 (one user per mesh row); a prefill call blocks the decode of the other rows (no interleaving); top_k / top_p are ignored by the sampled
spec round (temperature only: generation_config default top_p 1, top_k -1); sampled rounds also serve the greedy rows of a mixed batch (temperature ~0).
"""
import json
import os
import sys

import torch
from loguru import logger

PAD_TOKEN = 120002
ROWS = 4
CHUNK = 512
MAX_MODEL_LEN = 1048576
PAGE = 512

try:
    from vllm_tt_plugin.platform import TTRequestError
except Exception:  # noqa: BLE001  (demo / unit tests without the plugin)

    class TTRequestError(ValueError):
        pass


def configured_spec_k():
    """Draft length of the server's ``speculative_config`` (0 = plain). The loader hands ``initialize_vllm_model`` no vLLM config: the server's own command line is read (sys.argv survives the engine-core
    spawn); HY4_SPEC_K is an explicit override."""
    env = os.environ.get("HY4_SPEC_K")
    if env not in (None, ""):
        return int(env)
    argv = sys.argv
    for i, a in enumerate(argv):
        flag, eq, val = a.partition("=")
        if flag.replace("_", "-") != "--speculative-config":
            continue
        if not eq:
            val = argv[i + 1] if i + 1 < len(argv) else "{}"
        try:
            cfg = json.loads(val)
        except ValueError:
            return 0
        if cfg.get("method") == "custom_class":
            return int(cfg.get("num_speculative_tokens", 0))
    return 0


_SPEC_K = configured_spec_k()
_SAMPLED = os.environ.get("HY4_SPEC_SAMPLED", "1") == "1"
_SPEC_CAP_ENV = int(
    os.environ.get("HY4_SPEC_CAP", "99")
)  # max drafts OFFERED per row (0 = the non-speculative stream through the same verify-shape path: the lossless gate reference)


def _spec_cap():
    """max drafts OFFERED per row. HY4_SPEC_CAP_FILE (live switch, read at every proposal): a file holding an integer overrides HY4_SPEC_CAP (0 = the non-speculative stream through the same verify-shape path)."""
    f = os.environ.get("HY4_SPEC_CAP_FILE")
    if f:
        try:
            with open(f) as fh:
                return int(fh.read().strip())
        except (OSError, ValueError):
            pass
    return _SPEC_CAP_ENV


class _Slots:
    """logical slot (the plugin's state slot) -> physical mesh row. ``remap`` applies the plugin's permutation: new slot i reads old slot remap[i]."""

    def __init__(self, n):
        self.n = n
        self.phys = list(range(n))

    def remap(self, remap):
        r = [int(x) for x in remap]
        assert sorted(r) == list(range(self.n)), r
        self.phys = [self.phys[x] for x in r]


def _col(x, n, default):
    """per-row list of a sampling param (a scalar / short list / None -> padded with default)."""
    if x is None:
        return [default] * n
    x = list(x) if isinstance(x, (list, tuple)) else [x]
    x = [default if v is None else v for v in x]
    return (x + [x[-1] if x else default] * n)[:n]


class _Step:
    """one submitted plain decode step: the physical rows of its logical rows at submission time."""

    def __init__(self, rows, W):
        self.rows, self.W = rows, W


class HYV4ForCausalLM:
    """tt-vllm-plugin model class (see the module docstring)."""

    decode_input_update_contract = 1

    model_capabilities = {
        "supports_prefix_caching": False,
        "supports_chunked_prefill": False,  # whole-prompt prefill calls
        "prefill_chunk_alignment": CHUNK,
        "supports_async_decode": _SPEC_K
        == 0,  # plain: split submit / read-back + resident inputs; spec: synchronous verify (--no-async-scheduling)
        "supports_sample_on_device": True,
        "supports_device_penalties": False,
        "supports_spec_decode": _SPEC_K > 0,
        "spec_requirements": ["device_propose"] if _SPEC_K > 0 else [],
        "spec_hidden_handoff": [],
        "fabric_config": None,  # opened by additional_config tt.fabric_config (FABRIC_1D_RING)
    }
    if _SPEC_K > 0 and _SAMPLED:
        model_capabilities["supports_sampled_verify"] = True

    # ---- construction ------------------------------------------------------------------------------------------------------------------------
    @classmethod
    def initialize_vllm_model(
        cls, hf_config, mesh_device, max_batch_size, max_seq_len, tt_data_parallel=1, optimizations=None, **kwargs
    ):
        mesh = tuple(mesh_device.shape)
        if mesh != (ROWS, 8):
            raise TTRequestError(f"Hy4-Preview needs the (4, 8) Blackhole galaxy mesh, got {mesh}")
        if tt_data_parallel != 1:
            raise TTRequestError("Hy4-Preview does not support data parallelism")
        if max_batch_size > ROWS:
            raise TTRequestError(
                f"Hy4 serving supports max_num_seqs <= {ROWS} (one user per mesh row), got {max_batch_size}"
            )
        if max_seq_len > MAX_MODEL_LEN or max_seq_len < 1:
            raise TTRequestError(f"Hy4 serving supports max_model_len <= {MAX_MODEL_LEN}, got {max_seq_len}")
        return cls(mesh_device, max_batch_size, max_seq_len)

    @classmethod
    def spec_plan(cls, vllm_config, max_num_seqs, requested_k):
        """Plugin admission of ``speculative_config``: the engine is built for K = the launch's draft length (a verify round is S = K + 1 rows per user, T = 4 (K + 1) <= 16 token rows on the mesh -> K <= 3)."""
        from vllm_tt_plugin.spec_decode import SpecPlan, SpecReject

        if int(max_num_seqs) > ROWS:
            return SpecReject(f"max_num_seqs {max_num_seqs} > {ROWS} mesh rows", (1, 2, 3))
        if int(requested_k) < 1 or int(requested_k) > 3:
            return SpecReject(
                "a verify round runs 4 x (K + 1) <= 16 token rows: num_speculative_tokens must be 1..3", (1, 2, 3)
            )
        if _SPEC_K != int(requested_k):
            return SpecReject(
                f"the engine is built for K={_SPEC_K} (HY4_SPEC_K / --speculative-config at the server start), requested {requested_k}",
                (_SPEC_K,) if _SPEC_K else (),
            )
        return SpecPlan(
            effective_k=int(requested_k),
            lanes_per_request=1,
            extra_bytes_per_seq=0,
            extra_bytes_per_token=0,
            accept_modes=("argmax_ids",),
            drafter_state="internal",
            supports_narrow_decode=True,  # a step with nothing to verify is an ordinary [B, 1] decode (served as a forced single-token round); it also lets sampled requests be admitted (supports_sampled_verify)
        )

    def __init__(self, mesh_device, max_num_seqs, max_seq_len, *, vllm_config=None):
        from models.demos.blackhole.hy4_preview.tt.hy4_serve_engine import Hy4Engine

        self.md = mesh_device
        self.max_num_seqs, self.max_seq_len = max_num_seqs, max_seq_len
        self.eng = Hy4Engine(mesh_device, max_num_seqs, max_seq_len, k=_SPEC_K)
        self.vocab = self.eng.vocab
        self.pages = self.eng.pages
        self.slots = _Slots(max_num_seqs)
        self.apc_hook = None
        if self.eng.pfx and os.environ.get("HY4_APC_HOOK", "0") == "1":
            # home-row hold / row-match admission of the plugin scheduler read the row-local chains through this hook (default off)
            try:
                from vllm_tt_plugin.prefix_hooks import set_prefix_hook

                from models.demos.blackhole.hy4_preview.tt.hy4_apc_hook import Hy4ApcHook

                self.apc_hook = Hy4ApcHook(self.eng, ROWS)
                set_prefix_hook(self.apc_hook)
                logger.info("Hy4 APC hook registered (peek_home over the row-local chains)")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Hy4 APC hook not registered: {e!r}")
        self.steps = 0
        self._active = [0] * ROWS
        self._variant, self._seeds, self._samp_set = "G", [0] * ROWS, False
        self._warned_topk = False
        self._ttnn = None

    # vLLM inspects this protocol (``is_text_generation_model``) while building the ModelConfig, BEFORE the plugin loads the model.
    def embed_input_ids(self, input_ids):
        raise NotImplementedError("Use the TT plugin prefill_forward/decode_forward interface")

    def forward(self, input_ids, positions):
        raise NotImplementedError("Use the TT plugin prefill_forward/decode_forward interface")

    def compute_logits(self, hidden_states):
        raise NotImplementedError("The Hy4 generator owns the LM head")

    @classmethod
    def get_max_tokens_all_users(
        cls, model_name="", num_devices=1, tt_data_parallel=1, max_model_len=None, max_num_seqs=None, **kwargs
    ):
        return int(max_model_len or MAX_MODEL_LEN) * int(max_num_seqs or 1)

    # ---- KV cache (accounting only) ---------------------------------------------------------------------------------------------------------
    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers):
        logger.info(
            f"Hy4 vLLM: kv_cache_shape {tuple(kv_cache_shape)} x {num_layers} layers is accounting only; the model owns the SCALED_FP8 pool "
            f"({self.pages} pages of {PAGE} tokens per mesh row)"
        )
        self.kv_cache = [(torch.empty(0), torch.empty(0)) for _ in range(num_layers)]
        return self.kv_cache

    # ---- prefill -----------------------------------------------------------------------------------------------------------------------------
    def prefill_forward(
        self,
        tokens,
        start_pos=None,
        prompt_lens=None,
        page_table=None,
        kv_cache=None,
        enable_trace=True,
        empty_slots=None,
        sampling_params=None,
        **kwargs,
    ):
        """tokens [B, T] (prompt from position 0), prompt_lens [B], empty_slots [B] = destination logical slot. Returns the first token ids [B] int32 (sampled on the host from the last prompt position's logits row)
        when ``sampling_params`` is given, else fp32 logits [B, 1, V] (host sampling)."""
        import secrets

        B = int(tokens.shape[0])
        starts = [0] * B if start_pos is None else [int(x) for x in start_pos]
        lens = [int(tokens.shape[1])] * B if prompt_lens is None else [int(x) for x in prompt_lens]
        if any(s != 0 for s in starts):
            raise TTRequestError(
                "Hy4 serving prefills whole prompts (start_pos must be 0: no chunked prefill, no prefix cache)"
            )
        if any(n < 1 for n in lens) or any(n > self.max_seq_len for n in lens):
            raise TTRequestError(f"prompt length {max(lens)} outside 1..max_model_len {self.max_seq_len}")
        slots = list(range(B)) if empty_slots is None else [int(s) for s in empty_slots]
        if len(set(slots)) != B:
            raise RuntimeError(f"duplicate prefill slots {slots}")
        phys = [self.slots.phys[s] for s in slots]
        if self.eng.pfx:
            # row-match admission (HY4_PFX): serve each new request on the free mesh row that retains the longest prefix of its prompt. Device state is per PHYSICAL row, so moving a logical slot to another free
            # row is a swap of two entries of the logical -> physical map (both rows are idle).
            want = self.eng.assign_rows([tokens[i, : lens[i]].numpy() for i in range(B)], phys)
            for i, s in enumerate(slots):
                q = want[i]
                if self.slots.phys[s] != q:
                    s2 = self.slots.phys.index(q)
                    self.slots.phys[s], self.slots.phys[s2] = self.slots.phys[s2], self.slots.phys[s]
            phys = [self.slots.phys[s] for s in slots]
        if self.apc_hook is not None and kwargs.get("request_ids"):
            self.apc_hook.note_prefill(kwargs["request_ids"], phys)
        prompts = {p: [int(x) for x in tokens[i, : lens[i]].tolist()] for i, p in enumerate(phys)}
        sampling = {}
        if sampling_params is not None:
            if any(bool(x) for x in _col(getattr(sampling_params, "enable_log_probs", None), B, False)):
                raise TTRequestError("Hy4 device sampling does not support logprobs")
            temps = _col(getattr(sampling_params, "temperature", None), B, 0.0)
            ks = _col(getattr(sampling_params, "top_k", None), B, -1)
            ps = _col(getattr(sampling_params, "top_p", None), B, 1.0)
            sds = _col(getattr(sampling_params, "seed", None), B, None)
            for i, p in enumerate(phys):
                sampling[p] = (
                    float(temps[i] or 0.0),
                    int(ks[i]),
                    float(ps[i]),
                    (int(sds[i]) & 0x7FFFFFFF) if sds[i] is not None else secrets.randbits(31),
                )
        else:
            raise TTRequestError(
                "Hy4 product server: host sampling is not available (the plugin must sample on device)"
            )
        rids = {p: str(r) for p, r in zip(phys, kwargs["request_ids"])} if kwargs.get("request_ids") else None
        first = self.eng.prefill(prompts, sampling, rids=rids)
        for p in phys:
            self._seeds[p] = sampling[p][3]
        self._samp_set = False  # plain mode: the decode reload contract re-uploads tokens / positions / sampling state after a prefill
        return torch.tensor([first[p] for p in phys], dtype=torch.int32).reshape(B)

    # ---- decode ------------------------------------------------------------------------------------------------------------------------------
    def decode_forward(
        self,
        tokens,
        start_pos,
        page_table=None,
        kv_cache=None,
        enable_trace=True,
        read_from_device=True,
        sampling_params=None,
        slot_remap=None,
        reload_inputs=True,
        reload_page_table=False,
        reload_sampling_params=False,
        reset_sampling_state=False,
        num_valid_drafts=None,
        accepted_counts=None,
        spec_mode=None,
        **kwargs,
    ):
        if slot_remap is not None:
            self.slots.remap(
                slot_remap
            )  # device state is per PHYSICAL row (nothing moves): the logical -> physical map is all a remap changes
        if spec_mode is not None or num_valid_drafts is not None:
            if not self.eng.spec:
                raise ValueError(
                    "a speculative verify call arrived but the server runs plain decode (no speculative_config)"
                )
            return self._verify(tokens, start_pos, num_valid_drafts, sampling_params)
        if self.eng.spec:
            return self._decode_narrow(tokens, start_pos, sampling_params)
        return self._decode_plain(
            tokens,
            start_pos,
            sampling_params,
            read_from_device,
            reload_inputs,
            reload_sampling_params,
            reset_sampling_state,
        )

    # -- spec ---------------------------------------------------------------------------------------------------------------------------------
    def _verify(self, tokens, start_pos, num_valid_drafts, sp):
        from vllm_tt_plugin.spec_decode import VerifyOutput

        eng, K = self.eng, self.eng.K
        W, n = int(tokens.shape[0]), int(tokens.shape[1])
        if n != K + 1:
            raise ValueError(f"verify block width {n} != 1 + K ({1 + K})")
        nv = num_valid_drafts.reshape(-1).tolist()
        sp0 = start_pos.reshape(W, -1)[:, 0].tolist()
        caps, block, rows = [0] * ROWS, {}, []
        for i in range(W):
            if sp0[i] < 0:
                continue
            ph = self.slots.phys[i]
            if not eng.live[ph]:
                raise RuntimeError(f"verify of logical slot {i} (mesh row {ph}) that was never prefilled")
            caps[ph] = max(0, min(int(nv[i]), K))
            block[ph] = (tokens[i].tolist(), int(sp0[i]))
            rows.append((i, ph))
        want, temps, seeds = False, list(eng.temp), list(eng.seeds)
        if sp is not None:
            T = _col(getattr(sp, "temperature", None), W, 0.0)
            ks = _col(getattr(sp, "top_k", None), W, -1)
            ps = _col(getattr(sp, "top_p", None), W, 1.0)
            sd = _col(getattr(sp, "seed", None), W, None)
            for i, ph in rows:
                temps[ph] = float(T[i] or 0.0)
                if sd[i] is not None:
                    seeds[ph] = int(sd[i]) & 0x7FFFFFFF
                if temps[ph] > 0:
                    want = True
                    if (int(ks[i]) > 1 or float(ps[i]) < 1.0) and not self._warned_topk:
                        self._warned_topk = True
                        logger.warning("Hy4 sampled spec round: top_k / top_p are ignored (temperature only)")
        st = eng.round(caps, want, temps, seeds, plugin_block=block)
        out = torch.zeros(W, n, dtype=torch.int32)
        for i, ph in rows:
            out[i] = torch.tensor(st[ph]["a"], dtype=torch.int32)
        self.steps += 1
        if self.steps % 50 == 1:
            ms = eng.t_round / max(eng.rounds, 1) * 1e3
            logger.info(
                f"HY4 spec round {self.steps}: rows={len(rows)} m={[st[ph]['m'] for _, ph in rows]} round_ms_avg={ms:.1f} sampled={want}"
            )
        return VerifyOutput(spec_mode="argmax_ids", argmax_ids=out)

    def _decode_narrow(self, tokens, start_pos, sp):
        """ordinary [B, 1] decode in a speculating launch (no row carries a draft): a forced single-token round (cap 0): the verify rows beyond row 0 are computed and rejected, the MTP state advances as in
        any round, the next drafts are handed out by the propose call that follows. Returns the ids [W, 1] int32 (greedy or sampled by the row's temperature).
        """
        if sp is None:
            raise TTRequestError(
                "Hy4 product server: host sampling (penalties, logit_bias, bad_words, logprobs) is not available"
            )
        W = int(tokens.shape[0])
        sp0 = start_pos.reshape(-1).tolist()
        K = self.eng.K
        nv = torch.zeros(W, dtype=torch.int32)
        blk = torch.zeros(W, K + 1, dtype=torch.int32)
        blk[:, 0] = tokens.reshape(W, -1)[:, 0]
        pos = torch.full((W, K + 1), -1, dtype=torch.int32)
        for i in range(W):
            if sp0[i] >= 0:
                pos[i] = torch.arange(sp0[i], sp0[i] + K + 1, dtype=torch.int32)
        out = self._verify(blk, pos, nv, sp).argmax_ids
        return out[:, :1].clone()

    def propose_draft_tokens(self, num_drafts, committed_tokens, committed_positions, accepted_counts, hidden=None):
        """Hand out the drafts the last round / the seed round produced (they live in the engine's mirror). A row whose committed last token / position is not the engine's pending token / position is offered
        nothing (its next verify reloads the plugin's block)."""
        from vllm_tt_plugin.spec_decode import DraftOutput

        eng, K = self.eng, self.eng.K
        W = int(committed_tokens.shape[0])
        ids = torch.zeros(W, int(num_drafts), dtype=torch.int32)
        nv = torch.zeros(W, dtype=torch.int32)
        ct, cp, cn = committed_tokens.tolist(), committed_positions.tolist(), accepted_counts.tolist()
        for i in range(W):
            if cp[i][0] < 0:
                continue
            ph = self.slots.phys[i]
            if not eng.live[ph]:
                continue
            j = max(int(cn[i]), 1) - 1
            if int(ct[i][j]) == eng.pend[ph] and int(cp[i][j]) == eng.pos[ph]:
                k = min(K, int(num_drafts), _spec_cap())
                ids[i, :k] = torch.tensor([max(int(x), 0) for x in eng.drafts[ph][:k]], dtype=torch.int32)
                nv[i] = k
            else:
                logger.warning(
                    f"HY4 propose: row {i} (mesh row {ph}) committed ({int(ct[i][j])}, pos {int(cp[i][j])}) != engine pending ({eng.pend[ph]}, pos {eng.pos[ph]}): no drafts"
                )
        return DraftOutput(draft_token_ids=ids, num_valid=nv)

    # -- plain --------------------------------------------------------------------------------------------------------------------------------
    def _decode_plain(
        self, tokens, start_pos, sp, read_from_device, reload_inputs, reload_sampling_params, reset_sampling_state
    ):
        import secrets

        if sp is None:
            raise TTRequestError(
                "Hy4 product server: host sampling (penalties, logit_bias, bad_words) is not available"
            )
        dl = self.eng.dl
        W = int(tokens.shape[0])
        if any(bool(x) for x in _col(getattr(sp, "enable_log_probs", None), W, False)):
            raise TTRequestError("Hy4 device sampling does not support logprobs")
        rows = [(i, self.slots.phys[i]) for i in range(W) if int(start_pos[i]) >= 0]
        if reload_inputs:
            tok, pos, act = [PAD_TOKEN] * ROWS, [0] * ROWS, [0] * ROWS
            for i, ph in rows:
                p = int(start_pos[i])
                if p >= self.pages * PAGE:
                    raise TTRequestError(f"position {p} beyond the {self.pages * PAGE} tokens of the Hy4 pool")
                tok[ph], pos[ph], act[ph] = int(tokens[i, 0]), p, 1
            dl.load_inputs(tok, pos, act)
            self._active = act
        if reload_sampling_params or reset_sampling_state or not self._samp_set:
            T, K, P = [0.0] * ROWS, [1] * ROWS, [1.0] * ROWS
            temps, ks, ps = (
                _col(getattr(sp, "temperature", None), W, 0.0),
                _col(getattr(sp, "top_k", None), W, -1),
                _col(getattr(sp, "top_p", None), W, 1.0),
            )
            sds = _col(getattr(sp, "seed", None), W, None)
            for i, ph in rows:
                T[ph], K[ph], P[ph] = float(temps[i]), int(ks[i]), float(ps[i])
                if sds[i] is not None:
                    self._seeds[ph] = int(sds[i]) & 0x7FFFFFFF
                elif reset_sampling_state or not self._samp_set:
                    self._seeds[ph] = self._seeds[ph] or secrets.randbits(31)
            self._variant = dl.set_sampling(T, K, P, self._seeds, active=self._active)
            self._samp_set = True
        dl.submit(self._variant)
        self.steps += 1
        step = _Step(rows, W)
        if read_from_device:
            h = dl.read_async()
            return self._tokens_tensor(step, h)
        return step

    def _tokens_tensor(self, step, handle, sync=True):
        import ttnn

        if sync:
            ttnn.synchronize_device(self.md)
        ts = ttnn.get_device_tensors(handle)
        out = torch.zeros(step.W, 1, dtype=torch.int32)
        for i, ph in step.rows:
            out[i, 0] = int(ttnn.to_torch(ts[ph * 8]).reshape(-1)[0])
        return out

    def read_decode_output(self, tt_out, async_read=True, sample_rows=None):
        import ttnn

        if not isinstance(tt_out, _Step):
            return tt_out, []
        h = self.eng.dl.read_async()
        return (tt_out, h), [ttnn.record_event(self.md, 0)]

    def process_decode_output_host(self, handles, is_tokens=True, sample_rows=None):
        if isinstance(handles, tuple) and len(handles) == 2 and isinstance(handles[0], _Step):
            step, h = handles
            return self._tokens_tensor(step, h, sync=False)
        return handles

    def release_request(self, slot):
        """the request of this logical slot finished: the engine parks its mesh row (pad token, position 0). The pool pages are rewritten by the next prefill into the row."""
        try:
            self.eng.release(self.slots.phys[int(slot)])
        except (IndexError, ValueError):
            pass

    # ---- warm-up -----------------------------------------------------------------------------------------------------------------------------
    def warmup_model_prefill(self, kv_cache=None, enable_trace=True, *args, **kwargs):
        self.eng.warm()
        self.slots = _Slots(self.max_num_seqs)

    def warmup_model_decode(self, *args, **kwargs):
        self.eng.warm()
        self.slots = _Slots(self.max_num_seqs)
