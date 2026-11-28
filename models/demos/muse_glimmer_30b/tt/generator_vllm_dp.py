# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""vLLM (vllm-tt-plugin) adapter for Muse Glimmer 30B with in-process data parallelism (no speculative decoding).

`dp` replicas of a 1 x TP model (e.g. 4 x TP8 on a BH Galaxy) are built on submeshes of the one mesh vLLM opens and
driven by `MuseGlimmerGenerator`: global slot u lives in replica u // B at local slot u % B (B = max_num_seqs / dp).

Serving environment:

    TT_MODEL_CLASS_OVERRIDES="TTMuseGlimmerForConditionalGeneration=models.demos.muse_glimmer_30b.tt.generator_vllm_dp:MuseGlimmerDPForCausalLM"
    MESH_DEVICE=BHGLX            # or (8,4)
    TT_DATA_PARALLEL=4
    HF_MODEL=/mnt/tt-data/Muse-Glimmer-30B
    vllm serve ... --max-num-seqs 128 --sample-on-device-mode all \\
        --additional-config '{"tt": {...}}'   (chunked prefill disabled)

Serve WITHOUT --data_parallel_size (one engine; the dp replicas live in-process) and without --override-tt-config. The
plugin never passes tt_data_parallel > 1 for this model and does not read TT_DATA_PARALLEL, so the adapter reads the
TT_DATA_PARALLEL env var itself (it wins when set and > 0); --max-num-seqs is the global batch (B = max_num_seqs / dp).

Optional: MUSE_VLLM_MAX_SEQ_LEN caps the served model context (aligned up to 256). 32 users x 131072 tokens per
replica may not fit DRAM; the cap bounds the per-replica KV allocation. Prompts must then fit the capped context.

Model-owned KV: slot u owns page row u of its replica's cache; vLLM block tables and block ids are bookkeeping only and
are ignored. Sampling parameters are one shared set per decode batch (the generator loads one set on the samplers);
rows that disagree raise NotImplementedError. `slot_remap` is applied to the plugin-slot -> engine-slot map only (no
KV moves), once the decode that carries it succeeds.
"""

import os
import time

import numpy as np
import torch
from loguru import logger

_BLOCK_SIZE = 32
_SEQ_ALIGN = 256
_MAX_CONTEXT = 131072
_SUPPORTED_TP = (4, 8)
_MAX_BATCH_PER_REPLICA = 32
_MAX_DEVICE_TOP_K = 32


def _resolve_dp(max_batch_size, tt_data_parallel=1, env=None):
    """Return (dp, global batch, per-replica batch). Env TT_DATA_PARALLEL wins when set and > 0."""
    env = os.environ if env is None else env
    dp = int(env.get("TT_DATA_PARALLEL", 0) or 0) or int(tt_data_parallel)
    n = int(max_batch_size)
    if dp < 1:
        raise ValueError(f"tt_data_parallel must be >= 1, got {dp}")
    if n % dp != 0:
        raise ValueError(f"max_num_seqs={n} must be divisible by tt_data_parallel={dp}")
    b = n // dp
    if not 1 <= b <= _MAX_BATCH_PER_REPLICA:
        raise ValueError(
            f"per-replica batch {b} (max_num_seqs={n} / dp={dp}) must be in [1, {_MAX_BATCH_PER_REPLICA}]; "
            f"launch with --max-num-seqs <= {dp * _MAX_BATCH_PER_REPLICA}"
        )
    return dp, n, b


def _resolve_dp(max_batch_size, tt_data_parallel=1, env=None):
    """Return (dp, global batch, per-replica batch). Env TT_DATA_PARALLEL wins when set and > 0."""
    env = os.environ if env is None else env
    dp = int(env.get("TT_DATA_PARALLEL", 0) or 0) or int(tt_data_parallel)
    n = int(max_batch_size)
    if dp < 1:
        raise ValueError(f"tt_data_parallel must be >= 1, got {dp}")
    if n % dp != 0:
        raise ValueError(f"max_num_seqs={n} must be divisible by tt_data_parallel={dp}")
    b = n // dp
    if not 1 <= b <= _MAX_BATCH_PER_REPLICA:
        raise ValueError(
            f"per-replica batch {b} (max_num_seqs={n} / dp={dp}) must be in [1, {_MAX_BATCH_PER_REPLICA}]; "
            f"launch with --max-num-seqs <= {dp * _MAX_BATCH_PER_REPLICA}"
        )
    return dp, n, b


def _round_up(x, m):
    return -(-x // m) * m


def _served_seq_len(max_model_len):
    """Model max_seq_len for a served max_model_len (aligned to 256), optionally capped by MUSE_VLLM_MAX_SEQ_LEN."""
    s = _round_up(int(max_model_len), _SEQ_ALIGN)
    if s > _MAX_CONTEXT:
        raise ValueError(
            f"max_model_len={max_model_len} needs a {s}-token model context (rounded up to {_SEQ_ALIGN}), above the "
            f"{_MAX_CONTEXT} the model supports; launch with --max-model-len <= {_MAX_CONTEXT}"
        )
    cap = os.environ.get("MUSE_VLLM_MAX_SEQ_LEN")
    if cap:
        capped = _round_up(int(cap), _SEQ_ALIGN)
        if capped < s:
            logger.warning(f"MUSE_VLLM_MAX_SEQ_LEN caps the served context {s} -> {capped}")
            s = capped
    return s


def _as_list(x):
    if x is None:
        return [None]
    if isinstance(x, (list, tuple)):
        return list(x)
    return np.asarray(x.detach().cpu() if torch.is_tensor(x) else x).reshape(-1).tolist()


class _DecodeTiming:
    """MUSE_VLLM_TIMING=1: mean decode_forward period / issue time / read time, logged every 100 steps."""

    EVERY = 100

    def __init__(self):
        self._prev_enter = None
        self.reset()

    def reset(self):
        self.n = 0
        self.n_gap = 0
        self.gap = self.issue = self.read = 0.0
        self.n_read = 0
        self.bd = {}

    def on_issue(self, t_enter, t_exit, bd):
        if self._prev_enter is not None:
            self.gap += t_enter - self._prev_enter
            self.n_gap += 1
        self._prev_enter = t_enter
        self.issue += t_exit - t_enter
        for k, v in (bd or {}).items():
            self.bd[k] = self.bd.get(k, 0.0) + v
        self.n += 1
        if self.n >= self.EVERY:
            self.flush()

    def on_read(self, dt):
        self.read += dt
        self.n_read += 1

    def flush(self):
        ms = 1000.0
        logger.info(
            "MUSE_TIMING decode steps={} gap_ms={:.2f} issue_ms={:.2f} read_ms={:.2f} breakdown_ms={}".format(
                self.n,
                self.gap / max(self.n_gap, 1) * ms,
                self.issue / self.n * ms,
                self.read / max(self.n_read, 1) * ms,
                {k: round(v / self.n * ms, 3) for k, v in self.bd.items()},
            )
        )
        self.reset()


class MuseGlimmerDPForCausalLM:
    # Explicit reload commands (reload_inputs / reload_page_table / reload_sampling_params / reset_sampling_state):
    # steady-state device-sampled decode keeps tokens and positions resident on device (no host input copies).
    decode_input_update_contract = 1
    # The scheduler's KV bookkeeping is one uniform group; the model owns the physical cache.
    _HYBRID_KV_CACHE_GROUPS_ENABLED = False

    model_capabilities = {
        "supports_sample_on_device": True,
        "max_device_top_k": _MAX_DEVICE_TOP_K,
        "supports_device_penalties": False,
        "supports_device_grammar": False,
        "supports_async_decode": True,
        "supports_prefix_caching": False,
        "supports_chunked_prefill": False,
    }

    def __init__(self, generator, kv_handles=None):
        self.generator = generator
        self.kv_handles = kv_handles
        self.n_slots = int(generator.global_batch_size)
        self.B = int(generator.B)
        self.dp = int(generator.dp)
        self.phys = list(range(self.n_slots))  # plugin slot -> generator slot (see the module docstring)
        self._loaded_sampling = None  # key of the sampling set loaded on every replica's sampler, None if unknown
        self._dev_sp = None  # cached device SamplingParams for the loaded set (rebuilt only on reload/reset commands)
        self._last_active = []  # active rows of the last submitted decode (for output extraction without start_pos)
        self._timing = _DecodeTiming() if os.environ.get("MUSE_VLLM_TIMING") == "1" else None

    @classmethod
    def _from_generator(cls, generator, kv_handles=None):
        """Build the adapter around an existing generator (host tests build it around a fake one)."""
        if kv_handles is None:
            kv_handles = [list(m.kv_caches) for m in generator.models]
        return cls(generator, kv_handles)

    # vLLM's model-registry inspection needs the upstream interface; execution goes through the TT plugin.
    def embed_input_ids(self, input_ids):  # pragma: no cover - protocol shim
        raise RuntimeError("Use the TT plugin's prefill/decode interface")

    def forward(self, input_ids, positions, **kwargs):  # pragma: no cover - protocol shim
        raise RuntimeError("Use the TT plugin's prefill/decode interface")

    def compute_logits(self, hidden_states, **kwargs):  # pragma: no cover - protocol shim
        raise RuntimeError("The generator owns the LM head")

    # ---- config-time hooks
    @classmethod
    def get_max_tokens_all_users(cls, *, max_model_len=None, max_num_seqs=1, **kwargs):
        """The model owns one cache row of max_model_len tokens per sequence, so the scheduler pool covers them all."""
        n = int(max_num_seqs)
        if n < 1:
            raise ValueError(f"{cls.__name__} needs max_num_seqs >= 1, got {max_num_seqs}")
        if max_model_len is None:
            return n * _MAX_CONTEXT
        _served_seq_len(max_model_len)  # validates the context
        return n * int(max_model_len)

    @classmethod
    def get_kv_cache_spec(cls, vllm_config):
        """Bookkeeping-only uniform KV spec (every layer a FullAttentionSpec so vLLM keeps one group)."""
        from vllm.v1.kv_cache_interface import FullAttentionSpec

        cache_config = vllm_config.cache_config
        if cache_config.block_size != _BLOCK_SIZE:
            raise ValueError(f"Muse uses {_BLOCK_SIZE}-token KV pages, got block_size={cache_config.block_size}")
        text_config = getattr(vllm_config.model_config.hf_config, "text_config", vllm_config.model_config.hf_config)
        common = dict(
            block_size=_BLOCK_SIZE,
            num_kv_heads=text_config.num_key_value_heads,
            head_size=text_config.head_dim,
            dtype=vllm_config.model_config.dtype,
        )
        return {f"model.layers.{i}.self_attn": FullAttentionSpec(**common) for i in range(len(text_config.layer_types))}

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device,
        max_batch_size,
        max_seq_len=None,
        tt_data_parallel=1,
        optimizations=None,
        **kwargs,
    ):
        dp, n, b = _resolve_dp(max_batch_size, tt_data_parallel)
        logger.info(
            f"[muse dp] effective dp={dp} (tt_data_parallel={tt_data_parallel}, env TT_DATA_PARALLEL="
            f"{os.environ.get('TT_DATA_PARALLEL')!r}), max_num_seqs={n}, batch/replica={b}"
        )
        if optimizations is not None:
            raise ValueError(f"{cls.__name__} has one precision policy; drop the `optimizations` setting")
        if max_seq_len is None:
            raise ValueError("max_seq_len (the vLLM max_model_len) is required")
        n_dev = int(mesh_device.get_num_devices())
        if n_dev % dp != 0 or n_dev // dp not in _SUPPORTED_TP:
            raise ValueError(f"{n_dev} devices with tt_data_parallel={dp} give TP {n_dev / dp}; supported TP: {_SUPPORTED_TP}")
        tp = n_dev // dp
        seq_len = _served_seq_len(max_seq_len)

        from models.demos.muse_glimmer_30b.tt.generator import MuseGlimmerGenerator
        from models.demos.muse_glimmer_30b.tt.mesh_utils import dp_submeshes
        from models.demos.muse_glimmer_30b.tt.model import MuseGlimmerModel
        from models.demos.muse_glimmer_30b.tt.model_config import MuseGlimmerModelArgs, checkpoint_dir

        ckpt = checkpoint_dir()
        logger.info(
            f"[muse dp] {ckpt}: dp={dp} x tp={tp}, batch/replica={b}, max_model_len={max_seq_len} -> "
            f"model max_seq_len={seq_len}"
        )
        text_config = getattr(hf_config, "text_config", hf_config)
        models = []
        for i, sm in enumerate(dp_submeshes(mesh_device, dp, tp)):
            args = MuseGlimmerModelArgs.from_checkpoint(
                ckpt, mesh_device=sm, max_batch_size=b, max_seq_len=seq_len, max_prefill_len=seq_len
            )
            if i == 0 and int(text_config.vocab_size) != int(args.vocab_size):
                logger.warning(f"hf_config vocab_size {text_config.vocab_size} != model vocab_size {args.vocab_size}")
            models.append(MuseGlimmerModel(args, sm))
        return cls._from_generator(MuseGlimmerGenerator(models, enable_trace=True))

    # ---- KV: the model owns it
    def allocate_kv_cache(self, *args, **kwargs):
        del args, kwargs
        return self.kv_handles

    def allocate_kv_cache_per_layer(self, per_layer_specs):
        del per_layer_specs  # vLLM's sizing is bookkeeping only
        return self.kv_handles

    # ---- warmup: the generator compiles on first use and captures its decode traces on the second
    def warmup_model_prefill(self, kv_cache=None, enable_trace=False, can_sample_on_device=False, **kwargs):
        return None

    def warmup_model_decode(self, kv_cache=None, enable_trace=False, **kwargs):
        return None

    # ---- sampling
    def _uniform_sampling(self, sp, rows):
        """Shared (temperature, top_k, top_p, seed) of the given rows of a TTSamplingParams, or None for greedy."""
        if sp is None or not rows:
            return None
        temp, topk, topp = _as_list(sp.temperature), _as_list(sp.top_k), _as_list(sp.top_p)
        seeds = _as_list(getattr(sp, "seed", None))

        def at(vals, i):
            return vals[i if len(vals) > 1 else 0]

        def key(i):
            t, k, p = float(at(temp, i)), int(at(topk, i)), float(at(topp, i))
            if t <= 0 or k == 1:  # greedy whatever the other fields say
                return (0.0, 1, 1.0)
            return (t, k, p)

        keys = {key(i) for i in rows}
        if len(keys) > 1:
            raise NotImplementedError("per-request sampling params differ; Muse DP adapter supports a uniform set")
        t, k, p = keys.pop()
        if t == 0.0:
            return None
        seed = seeds[rows[0] if len(seeds) > 1 else 0]
        return {"temperature": t, "top_k": k, "top_p": p, "seed": seed}

    def _device_params(self, sampling):
        """SamplingParams for the generator from the uniform dict (None -> greedy)."""
        gen = self.generator
        seed = None if sampling is None else sampling["seed"]
        return gen._to_sampling_params(sampling, gen.models[0].args.max_top_k, seed)

    # ---- prefill / decode
    def prefill_forward(
        self,
        tokens,
        page_table=None,
        kv_cache=None,
        prompt_lens=None,
        start_pos=None,
        empty_slots=None,
        sampling_params=None,
        enable_trace=True,
        page_tables_per_layer=None,
        **kwargs,
    ):
        tokens = torch.as_tensor(tokens)
        n_rows = int(tokens.shape[0])
        if empty_slots is None:
            raise ValueError(f"{type(self).__name__}.prefill_forward needs empty_slots (the global slot of each row)")
        slots = [int(s) for s in _as_list(empty_slots)]
        if len(slots) != n_rows or any(not 0 <= s < self.n_slots for s in slots):
            raise ValueError(f"empty_slots {slots} do not name {n_rows} of the {self.n_slots} slots")
        if start_pos is not None and any(int(x) != 0 for x in np.asarray(start_pos).reshape(-1).tolist()):
            raise ValueError(
                "chunked, resumed or prefix-cached prefill is not supported (start_pos must be 0); "
                "disable chunked prefill (--no-enable-chunked-prefill) and prefix caching"
            )
        for name in ("pixel_values", "pixel_values_videos"):
            if kwargs.get(name) is not None and any(v is not None for v in kwargs[name]):
                raise ValueError(f"{type(self).__name__} is text-only; {name} is not supported")
        pl = None if prompt_lens is None else _as_list(prompt_lens)
        lens = [int(pl[r]) if pl is not None else int(tokens.shape[1]) for r in range(n_rows)]
        max_seq_len = int(self.generator.models[0].args.max_seq_len)
        for n in lens:
            if not 1 <= n < max_seq_len:
                raise ValueError(f"prompt of {n} tokens does not fit the model context {max_seq_len}")

        dev_sp = None
        if sampling_params is not None:
            dev_sp = self._device_params(self._uniform_sampling(sampling_params, list(range(n_rows))))
        # The samplers' parameters change (prefill loads them per replica); decode reloads them on every replica.
        self._loaded_sampling = None
        out = []
        for r, (s, n) in enumerate(zip(slots, lens)):
            ids = [int(t) for t in tokens[r, :n].tolist()]
            out.append(self.generator.prefill_slot(self.phys[s], ids, dev_sp))
        if dev_sp is not None:
            return torch.tensor(out, dtype=torch.int64).reshape(n_rows, 1), None
        return torch.stack([o.float().reshape(1, -1) for o in out])  # host logits [N, 1, V]

    def _remapped(self, slot_remap):
        """phys after `slot_remap` (slot i now names the state that slot slot_remap[i] named)."""
        if slot_remap is None:
            return list(self.phys)
        idx = [int(j) for j in _as_list(slot_remap)]
        if sorted(idx) != list(range(self.n_slots)):
            raise RuntimeError(f"slot_remap {idx} is not a permutation of the {self.n_slots} slots")
        return [self.phys[j] for j in idx]

    def decode_forward(
        self,
        tokens,
        page_table=None,
        kv_cache=None,
        start_pos=None,
        enable_trace=True,
        read_from_device=True,
        sampling_params=None,
        slot_remap=None,
        reset_batch=None,
        prompt_tokens=None,
        output_tokens=None,
        page_tables_per_layer=None,
        reload_inputs=True,
        reload_page_table=False,
        reload_sampling_params=False,
        reset_sampling_state=False,
        **kwargs,
    ):
        """Issue one decode step. reload_inputs=False: tokens/positions stay device-resident (previous step's sampled
        token fed back on device); only device-sampled steps may do that. Returns (handle, is_sample) unread when
        read_from_device is False, else the host result of process_decode_output_host."""
        # reset_batch / prompt_tokens / output_tokens only matter for device penalties (unsupported).
        if (reload_inputs and reload_page_table) or (reset_sampling_state and not reload_inputs):
            raise ValueError("Invalid v1 decode reload commands")
        if slot_remap is not None and not reload_inputs:
            raise ValueError("A slot remap requires authoritative input reconstruction")
        tm = self._timing
        t_enter = time.perf_counter() if tm is not None else 0.0
        gen = self.generator
        sample = sampling_params is not None
        if not sample and not reload_inputs:
            raise ValueError("host-sampled (logits) decode requires reload_inputs=True")
        sp = None if start_pos is None else torch.as_tensor(start_pos).reshape(-1)
        if reload_inputs:
            tokens = torch.as_tensor(tokens).reshape(-1)
            if sp is None:
                raise ValueError("start_pos is required when reload_inputs is set")
            rows = int(tokens.shape[0])
            if rows != self.n_slots or int(sp.shape[0]) != self.n_slots:
                raise ValueError(f"decode batch must have {self.n_slots} rows, got {rows} tokens / {sp.shape[0]}")
        phys = self._remapped(slot_remap)  # slot_remap is only legal with reload_inputs
        if sp is not None and int(sp.shape[0]) == self.n_slots:
            active = [i for i in range(self.n_slots) if int(sp[i]) >= 0]
        else:
            active = list(self._last_active)

        dev_sp, load = None, None
        if sample:
            need_eval = (
                reload_sampling_params or reset_sampling_state or self._dev_sp is None or self._loaded_sampling is None
            )
            if need_eval:
                sampling = self._uniform_sampling(sampling_params, active)
                key = ("greedy",) if sampling is None else tuple(sorted(sampling.items(), key=lambda kv: kv[0]))
                self._dev_sp = self._device_params(sampling)
                if reload_sampling_params or reset_sampling_state or key != self._loaded_sampling:
                    load = self._dev_sp
                self._loaded_sampling = key
            dev_sp = self._dev_sp

        copy_pt = bool(reload_page_table or getattr(gen, "page_table_dirty", False))
        if reload_inputs:
            # Row i's state lives in generator slot phys[i]; gaps are position -1 / token 0 as in generate().
            g_tokens = torch.zeros(self.n_slots, dtype=torch.int64)
            g_pos = torch.full((self.n_slots,), -1, dtype=torch.int64)
            for i in active:
                g_tokens[phys[i]] = int(tokens[i])
                g_pos[phys[i]] = int(sp[i])
            handle = gen.decode_issue(
                g_tokens, g_pos, sampling=dev_sp, sampling_params=load, reload=True, copy_page_table=copy_pt
            )
        else:
            handle = gen.decode_issue(
                None, None, sampling=dev_sp, sampling_params=load, reload=False, copy_page_table=copy_pt
            )
        self.phys = phys
        self._last_active = active
        result = ({"handle": handle, "phys": phys, "active": active}, sample)
        if tm is not None:
            tm.on_issue(t_enter, time.perf_counter(), getattr(gen, "last_step_breakdown", None))
        if read_from_device:
            return self.process_decode_output_host(self.read_decode_output(result), is_tokens=sample)
        return result

    def read_decode_output(self, tt_out, async_read=False, sample_rows=None):
        """Read a submitted decode. Returns (wrapped, is_tokens), plus per-replica events when async_read.

        sample_rows is accepted for the plugin's compact-readback protocol and ignored (the full small token vector,
        or the full logits, is read)."""
        del sample_rows
        wrapped, is_tokens = tt_out
        t0 = time.perf_counter() if self._timing is not None else 0.0
        payload, events = self.generator.decode_read(wrapped["handle"], async_read=async_read)
        wrapped = dict(wrapped, payload=payload)
        result = (wrapped, is_tokens)
        if self._timing is not None:
            self._timing.on_read(time.perf_counter() - t0)
        return (result, events) if async_read else result

    def process_decode_output_host(self, tt_out, is_tokens=False, sample_rows=None):
        """Slot-ordered host result: int64 tokens [rows, 1] (is_tokens) or float32 logits [rows, 1, V]; gap rows zero.
        Returns (tensor, None)."""
        del sample_rows
        wrapped, sampled = tt_out
        if "payload" not in wrapped:  # synchronous path hands over the unread submission
            wrapped, sampled = self.read_decode_output(tt_out)
        res = self.generator.decode_finish(wrapped["handle"], wrapped["payload"])  # generator-slot order
        phys, active = wrapped["phys"], wrapped["active"]
        rows = self.n_slots
        if sampled:
            out = torch.zeros(rows, 1, dtype=torch.int64)
            for i in active:
                out[i, 0] = int(res[phys[i]])
        else:
            res = res.float()
            out = torch.zeros(rows, 1, res.shape[-1], dtype=torch.float32)
            for i in active:
                out[i, 0] = res[phys[i]]
        return out, None

    # ---- lifecycle
    def release_request(self, slot):
        """Nothing to free: the slot's KV row is overwritten by the next prefill and positions come from vLLM."""
        del slot

    def note_state_slots_moved(self, moves):
        """decode_forward already applied the matching slot_remap to `phys`; nothing moves on the device."""
        del moves

    def release_persistent_capture(self):
        self.generator.release_traces()
