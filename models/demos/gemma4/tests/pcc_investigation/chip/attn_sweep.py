# Sweep attention settings (precision AND order of computation) for accuracy, on the QB2, building the model once.
# Each setting patches the attention calls at runtime (accuracy runs are untraced, so every call sees the patch) and runs
#   gate: the accuracy gate's text (Declaration of Independence), prefill 128 + 128 teacher-forced decode steps
#   book: A Tale of Two Cities, prefill 512 + 499 teacher-forced decode steps (tt-metal's token-accuracy protocol)
# scored per position against Hugging Face fp32 (gate_hf_fp32_logits.pt, book_logits_hf_fp32.pt) and bf16 (the gate's
# own cached reference / the .refpt logits). Results append to $GEMMA4_PCC_DATA/attn_sweep_<label>.jsonl as they finish.
# Usage: attn_sweep.py <label> [setting names...]    (no names: all settings in SETTINGS order)
import os as _os
from pathlib import Path as _Path

DATA = _os.environ.get("GEMMA4_PCC_DATA", _os.path.expanduser("~/benchmark-data/gemma4-pcc"))
MODELS = _os.environ.get("GEMMA4_PCC_MODELS", _os.path.expanduser("~/benchmark-data"))
REPO = _os.environ.get("GEMMA4_PCC_REPO") or str(_Path(__file__).resolve().parents[6])

import gc, json, math, os, sys, time, traceback
from pathlib import Path

import torch
import ttnn
from models.demos.gemma4.tt import fp32_mode
from models.demos.gemma4.tt.attention import decode as DEC
from models.demos.gemma4.tt.attention import prefill as PRE
from models.tt_transformers.tests.optimizer_weight_cache import RunCache

LABEL = sys.argv[1]
ONLY = sys.argv[2:]
MODEL = f"{MODELS}/gemma-4-26B-A4B-it"
D = Path(DATA)
OUT = D / f"attn_sweep_{LABEL}.jsonl"

# ---------------------------------------------------------------- settings
F = ttnn.MathFidelity


def ckc(fid, approx=False, fp32=False, l1=False):
    return ttnn.types.BlackholeComputeKernelConfig(math_fidelity=fid, math_approx_mode=approx, fp32_dest_acc_en=fp32, packer_l1_acc=l1)


SETTINGS = {
    "baseline": {},
    # SDPA decode: precision (default compute config = HiFi2, math_approx_mode=True, no fp32 accumulation)
    "sdpa_hifi4_exact": {"sdpa_ckc": (F.HiFi4, False, False, False)},
    "sdpa_hifi2_exact": {"sdpa_ckc": (F.HiFi2, False, False, False)},
    "sdpa_hifi4_approx": {"sdpa_ckc": (F.HiFi4, True, False, False)},
    "sdpa_exp_approx": {"exp_approx": True},
    # SDPA decode: order (keys per online-softmax step; cores sharing one head, merged by a reduction tree)
    "sdpa_kchunk32": {"k_chunk": 32},
    "sdpa_kchunk128": {"k_chunk": 128},
    "sdpa_kchunk256": {"k_chunk": 256},
    "sdpa_cores_per_head1": {"cores_per_head": 1},
    "sdpa_cores_per_head4": {"cores_per_head": 4},
    # QKV matmul: precision (sliding layers run LoFi: program config without compute config) and K blocking
    "qkv_hifi2": {"qkv": dict(fid=F.HiFi2, fp32=True)},
    "qkv_hifi4": {"qkv": dict(fid=F.HiFi4, fp32=True)},
    "qkv_default_config": {"qkv": dict(default_pc=True)},
    "qkv_hifi4_kblock1": {"qkv": dict(fid=F.HiFi4, fp32=True, in0_block_w=1)},
    "qkv_hifi4_kblock11": {"qkv": dict(fid=F.HiFi4, fp32=True, in0_block_w=11)},
    "qkv_lofi_kblock1": {"qkv": dict(fid=F.LoFi, fp32=False, in0_block_w=1)},
    # output projection (HiFi2 + fp32 accumulation now)
    "oproj_hifi4": {"oproj": dict(fid=F.HiFi4, fp32=True)},
    "oproj_hifi4_kblock1": {"oproj": dict(fid=F.HiFi4, fp32=True, in0_block_w=1)},
    "oproj_hifi4_kblock16": {"oproj": dict(fid=F.HiFi4, fp32=True, in0_block_w=16)},
    # per-head Q/K/V RMSNorm in decode (op default now; prefill already HiFi4 + fp32)
    "headnorm_decode_fp32": {"headnorm_fp32": True},
    # all-reduce order: reduce-scatter + all-gather instead of ttnn.all_reduce
    "ccl_async": {"env": {"GEMMA4_CCL_ASYNC": "1"}},
    # prefill SDPA chunking (writes the KV cache decode reads); now q=256/k=128 sliding, 128/128 global
    "prefill_kchunk32": {"env": {"GEMMA4_PREFILL_SDPA_KCHUNK": "32"}},
    "prefill_kchunk256": {"env": {"GEMMA4_PREFILL_SDPA_KCHUNK": "256"}},
    "prefill_qchunk32": {"env": {"GEMMA4_PREFILL_SDPA_QCHUNK": "32"}},
    "prefill_qchunk128": {"env": {"GEMMA4_PREFILL_SDPA_QCHUNK": "128"}},
    # prefill QKV matmul (default config = HiFi2) at HiFi4 + fp32 accumulation
    "prefill_qkv_hifi4": {"prefill_qkv": dict(fid=F.HiFi4, fp32=True)},
    "baseline_again": {},
    # combinations
    "combo_qkv_hifi4_kchunk32": {"qkv": dict(fid=F.HiFi4, fp32=True), "k_chunk": 32},
    "combo_max_precision": {"qkv": dict(fid=F.HiFi4, fp32=True), "sdpa_ckc": (F.HiFi4, False, False, False),
                            "headnorm_fp32": True, "oproj": dict(fid=F.HiFi4, fp32=True)},
    "combo_max_precision_kchunk32": {"qkv": dict(fid=F.HiFi4, fp32=True), "sdpa_ckc": (F.HiFi4, False, False, False),
                                     "headnorm_fp32": True, "oproj": dict(fid=F.HiFi4, fp32=True), "k_chunk": 32},
}
SET = {}

# ---------------------------------------------------------------- patches
_sdpa = ttnn.transformer.paged_scaled_dot_product_attention_decode


def sdpa(*a, **k):
    pc = k.get("program_config")
    if pc is not None and any(x in SET for x in ("k_chunk", "exp_approx", "cores_per_head")):
        k["program_config"] = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=pc.compute_with_storage_grid_size,
            q_chunk_size=pc.q_chunk_size,
            k_chunk_size=SET.get("k_chunk", pc.k_chunk_size),
            exp_approx_mode=SET.get("exp_approx", pc.exp_approx_mode),
            max_cores_per_head_batch=SET.get("cores_per_head", pc.max_cores_per_head_batch),
        )
    if "sdpa_ckc" in SET:
        k["compute_kernel_config"] = ckc(*SET["sdpa_ckc"])
    return _sdpa(*a, **k)


ttnn.transformer.paged_scaled_dot_product_attention_decode = sdpa


def _linear_1d(x, w, s, memory_config=None):
    """Decode (M=32) linear with the swept precision / K blocking; 1D mcast over an 8x6 grid like the tuned configs."""
    if s.get("default_pc"):
        return ttnn.linear(x, w, memory_config=memory_config)
    k_tiles = int(x.shape[-1]) // 32
    n_tiles = (int(w.shape[-1]) + 31) // 32
    kb = s.get("in0_block_w", 4)
    assert k_tiles % kb == 0, (k_tiles, kb)
    pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(8, 6), in0_block_w=kb, out_subblock_h=1, out_subblock_w=2,
        per_core_M=1, per_core_N=max(2, (n_tiles + 47) // 48), fuse_batch=True, fused_activation=None, mcast_in0=True)
    return ttnn.linear(x, w, memory_config=memory_config, program_config=pc,
                       compute_kernel_config=ckc(s.get("fid", F.LoFi), False, s.get("fp32", False), True))


_qkv = DEC.apply_qkv_projection


def qkv(hidden_states, weights, memory_config=None):
    s = SET.get("qkv")
    if s is None or hidden_states.padded_shape[-2] != 32:
        return _qkv(hidden_states, weights, memory_config)
    if memory_config is None and weights.wqkv.shape[-1] == 2048:
        memory_config = ttnn.L1_MEMORY_CONFIG  # as the tuned sliding-layer path writes it
    return _linear_1d(hidden_states, weights.wqkv, s, memory_config)


DEC.apply_qkv_projection = qkv

_oproj = DEC.apply_output_projection


def oproj(tensor, weights):
    s = SET.get("oproj")
    if s is None:
        return _oproj(tensor, weights)
    out = _linear_1d(tensor, weights.o_proj, s)
    tensor.deallocate(True)
    return out


DEC.apply_output_projection = oproj

_headnorm = DEC.apply_per_head_norm


def headnorm(*a, **k):
    if SET.get("headnorm_fp32"):
        k["fp32_accumulate"] = True
    return _headnorm(*a, **k)


DEC.apply_per_head_norm = headnorm

_pqkv = PRE.apply_qkv_projection


def pqkv(hidden_states, weights, memory_config=None):
    s = SET.get("prefill_qkv")
    if s is None:
        return _pqkv(hidden_states, weights, memory_config)
    return ttnn.linear(hidden_states, weights.wqkv, memory_config=memory_config,
                       compute_kernel_config=ckc(s["fid"], False, s.get("fp32", False), False))


PRE.apply_qkv_projection = pqkv

# ---------------------------------------------------------------- isolated per-layer error (SWEEP_ISO=<positions>)
# Decode positions 0..N-1 of the book one at a time with every layer fed Hugging Face's exact fp32 input
# (hf_per_layer_ref_512.pt), so each layer's own error is measured without the cascade from earlier layers.
ISO_N = int(os.environ.get("SWEEP_ISO", "0"))
ISO = {"on": False, "pos": 0, "layer": None, "mesh": None}
REC = {}
if ISO_N:
    from models.demos.gemma4.tt import layer as _L, router as _R, shared_mlp as _SM
    from models.demos.gemma4.tt.attention import Gemma4Attention as _ATT
    from models.demos.gemma4.tt.experts import Gemma4Experts as _EXP

    PL_REF = torch.load(D / "hf_per_layer_ref_512.pt")

    def _put(kind, t):
        x = ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()
        REC.setdefault(kind, {}).setdefault(ISO["layer"], []).append(x.reshape(-1, x.shape[-1])[:1])

    _lcall = _L.Gemma4DecoderLayer.__call__

    def _layer_call(self, hidden_states, *a, **k):
        if not ISO["on"]:
            return _lcall(self, hidden_states, *a, **k)
        ISO["layer"] = self.layer_idx
        shape = list(hidden_states.shape)
        x = torch.zeros(shape)
        x.reshape(-1, shape[-1])[0] = PL_REF["in"][self.layer_idx][ISO["pos"]]
        x = x if hidden_states.dtype == ttnn.float32 else x.to(torch.bfloat16)
        hidden_states = ttnn.from_torch(x, device=ISO["mesh"], dtype=hidden_states.dtype, layout=hidden_states.layout,
                                        mesh_mapper=ttnn.ReplicateTensorToMesh(ISO["mesh"]), memory_config=hidden_states.memory_config())
        out = _lcall(self, hidden_states, *a, **k)
        _put("out", out)
        return out

    _L.Gemma4DecoderLayer.__call__ = _layer_call
    for _cls, _kind in ((_ATT, "attn"), (_EXP, "experts")):
        def _wrapped(self, *a, _orig=_cls.__call__, _kind=_kind, **k):
            out = _orig(self, *a, **k)
            if ISO["on"]:
                _put(_kind, out)
            return out
        _cls.__call__ = _wrapped
    _rcall = _R.Gemma4Router.__call__

    def _router_call(self, hidden_states):
        out = _rcall(self, hidden_states)
        if ISO["on"]:
            _put("routing", out)
        return out

    _R.Gemma4Router.__call__ = _router_call


def iso_score():
    """Mean over the 30 layers: error (1 - PCC) of the attention output and layer output, and same-8-experts share."""
    res = {}
    n = len(REC["out"][0])
    for kind in ("attn", "experts", "out"):
        errs = []
        for i in sorted(REC[kind]):
            a = torch.cat(REC[kind][i]).double(); b = PL_REF[kind][i][:n].double()
            a = a - a.mean(-1, keepdim=True); b = b - b.mean(-1, keepdim=True)
            errs.append((1 - ((a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1)))).mean().item())
        res[f"{kind}_err_e4"] = round(sum(errs) / len(errs) * 1e4, 3)  # 1 - PCC, in units of 1e-4
    same = []
    for i in sorted(REC["routing"]):
        tt = torch.cat(REC["routing"][i])[:, :128] != 0
        hf = torch.zeros_like(tt); hf.scatter_(1, PL_REF["idx"][i][:n], True)
        same.append(((tt & hf).sum(-1) == 8).float().mean().item())
    res["same8_pct"] = round(sum(same) / len(same) * 100, 2)
    return res

# ---------------------------------------------------------------- references
from transformers import AutoTokenizer
from models.demos.gemma4.tests import test_optimizer_gemma4_pcc as GATE

tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
gate_tokens = GATE.encode_text(tok)[: GATE.PROMPT_TOKENS + GATE.FORCED_TOKENS]
os.environ.setdefault("HF_MODEL", MODEL)
gate_bf16 = GATE._reference_logits(MODEL, gate_tokens).float()
gate_f32 = torch.load(D / "gate_hf_fp32_logits.pt").float()
assert gate_f32.shape[0] == len(gate_tokens), (gate_f32.shape, len(gate_tokens))
refpt = torch.load(D / "gemma-4-26B-A4B-it.refpt")
book_tokens = refpt["reference_tokens"][0]
book_f32 = torch.load(D / "book_logits_hf_fp32.pt").float()
book_bf16 = torch.load(D / "gemma-4-26B-A4B-it.refpt.logits.pt").float()


def pcc_rows(a, b):
    a = a.double() - a.double().mean(-1, keepdim=True)
    b = b.double() - b.double().mean(-1, keepdim=True)
    return (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1))


def score(L, f32, bf16):
    pf, pb = pcc_rows(L, f32), pcc_rows(L, bf16)
    ref = bf16.argmax(-1)
    top5 = L.topk(5, -1).indices
    return dict(pcc_fp32=round(pf.mean().item(), 5), pcc_bf16=round(pb.mean().item(), 5),
                min_pcc_fp32=round(pf.min().item(), 4), below_099_fp32=int((pf < 0.99).sum()),
                top1_bf16=round((L.argmax(-1) == ref).float().mean().item() * 100, 2),
                top5_bf16=round((top5 == ref[:, None]).any(1).float().mean().item() * 100, 2),
                first_pcc_fp32=round(pf[0].item(), 5))


# ---------------------------------------------------------------- model
os.environ.setdefault("HF_HUB_OFFLINE", "1")
cache = RunCache(Path(f"{REPO}/generated/optimizer_cache"), f"gemma4-attnsweep-{LABEL}-")
os.environ["TT_CACHE_PATH"] = cache.path
mesh = generator = kv = None
try:
    from models.demos.gemma4.tt.generator import Gemma4Generator
    from models.demos.gemma4.tt.generator_trace import resolve_gemma4_demo_long_context
    from models.demos.gemma4.tt.model_config import Gemma4ModelArgs
    from models.tt_transformers.tt.common import PagedAttentionConfig

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 4), l1_small_size=24576, num_command_queues=1)
    ISO["mesh"] = mesh
    pac = PagedAttentionConfig(block_size=32, max_num_blocks=math.ceil(1024 / 32))
    lc = resolve_gemma4_demo_long_context(1024, mesh, MODEL, paged_attention=True)
    generator, kv, _ = cache.build(
        lambda: Gemma4Generator.from_pretrained(mesh_device=mesh, model_path=MODEL, max_batch_size=1, max_seq_len=1024,
                                                paged_attention_config=pac, bounded_sliding_kv_cache=lc["bounded_sliding"]),
        loaders=[(Gemma4ModelArgs, "load_state_dict")])
    gc.collect()
    cache.loaded()
    vocab = generator.model_args[0].vocab_size
    page_table = torch.arange(pac.max_num_blocks, dtype=torch.int32).reshape(1, pac.max_num_blocks)

    def host(out):
        first = out[0] if isinstance(out, (tuple, list)) else out
        if not isinstance(first, torch.Tensor):
            first = GATE._vocab_logits(first, vocab)
        return first.float().reshape(-1, first.shape[-1])[0, :vocab]

    def run(tokens, n_prompt, n_total):
        rows = [host(generator.prefill_forward_text(torch.as_tensor(tokens[:n_prompt]).reshape(1, -1).long(), page_table=page_table,
                                                    kv_cache=kv, prompt_lens=[n_prompt], warmup_prefill=False, enable_trace=False,
                                                    sampling_params=None))]
        for pos in range(n_prompt, n_total):
            rows.append(host(generator.decode_forward(torch.tensor([[int(tokens[pos])]], dtype=torch.long), torch.tensor([pos], dtype=torch.int64),
                                                      page_table=page_table, kv_cache=kv, enable_trace=False, sampling_params=None,
                                                      reload_inputs=True, reload_page_table=False, reload_sampling_params=False,
                                                      reset_sampling_state=False)))
        return torch.stack(rows)  # positions n_prompt-1 .. n_total-1

    names = ONLY or list(SETTINGS)
    for name in names:
        s = SETTINGS[name]
        SET.clear(); SET.update({k: v for k, v in s.items() if k != "env"})
        saved_env = {k: os.environ.get(k) for k in s.get("env", {})}
        os.environ.update(s.get("env", {}))
        r = {"setting": name}
        t = time.time()
        try:
            g = run(gate_tokens, GATE.PROMPT_TOKENS, len(gate_tokens))
            r["gate"] = score(g, gate_f32[GATE.PROMPT_TOKENS - 1:], gate_bf16[GATE.PROMPT_TOKENS - 1:])
            t_book = time.time()
            b = run(book_tokens, 512, 1011)
            r["book"] = score(b, book_f32[511:1011], book_bf16[511:1011])
            r["book_s_per_decode"] = round((time.time() - t_book) / 499, 3)
            if ISO_N:
                REC.clear(); ISO["on"] = True
                try:
                    for pos in range(ISO_N):
                        ISO["pos"] = pos
                        generator.decode_forward(torch.tensor([[int(PL_REF["ids"][pos])]], dtype=torch.long), torch.tensor([pos], dtype=torch.int64),
                                                 page_table=page_table, kv_cache=kv, enable_trace=False, sampling_params=None,
                                                 reload_inputs=True, reload_page_table=False, reload_sampling_params=False,
                                                 reset_sampling_state=False)
                finally:
                    ISO["on"] = False
                r["iso"] = iso_score()
        except Exception as e:  # a setting the op refuses: record it and go on
            r["error"] = f"{type(e).__name__}: {str(e)[:300]}"
            traceback.print_exc()
        finally:
            for k, v in saved_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        r["seconds"] = round(time.time() - t)
        print("SWEEP " + json.dumps(r), flush=True)
        with open(OUT, "a") as f:
            f.write(json.dumps(r) + "\n")
    print("DONE", flush=True)
finally:
    generator = kv = None
    gc.collect()
    if mesh is not None:
        ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    cache.cleanup()
