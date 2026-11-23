# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""DP4 x TP8 cold prefill for Llama-3.3-70B on the 8x4 Galaxy (LLAMA_BH_PREFILL_DP4=1).

Why: the TP32 prefill splits ONE prompt over all 32 chips. Per 4K layer it moves ~1.7 ms of all-gather /
reduce-scatter traffic against ~0.95 ms of matmul, so it is communication-bound and no kernel tuning reaches the
batch-32 targets. Here every mesh column (8 chips, cluster_axis 0) prefills ITS OWN lane's prompt with TP8, so the
four prompts of a vLLM prefill step (one per lane: users 0-7 live on column 0, 8-15 on column 1, ...) run at once on
disjoint links, and each chip does 4x the matmul work per collective byte.

Layout per chip (r, c): the residual is sequence-parallel, rows r*M/8 .. (r+1)*M/8 of prompt c, full hidden 8192.
  RMSNorm (local rows) -> all-gather rows (bf8) -> QKV [8192, 1280] (8 q heads 8r..8r+7 + kv head r)
  -> heads + RoPE -> paged_fill_cache (kv head r, prompt c: exactly where decode reads it) -> causal SDPA
  -> WO [1024, 8192] + reduce-scatter rows (matmul_reduce_scatter_prefill) -> + residual
  -> RMSNorm -> all-gather rows -> FF1|FF3 packed [8192, 7168] -> silu * mul -> FF2 [3584, 8192] + reduce-scatter
  -> + residual
The RMSNorm gammas are folded into the QKV / FF1|FF3 weights; the norm itself is x * rsqrt(mean(x^2) + eps) with the
mean taken by a matmul against a ones column (ttnn.rms_norm over a full 8192-wide row overflows L1).
Precision matches the TP32 prefill: bf8 QKV / WO / FF2 at HiFi2, bf4 FF1|FF3 at LoFi, HiFi2 SDPA, bf16 residual.
Output: each column's last-token hidden row, handed to the model's own final norm / LM head / sampling.
Weights: a second copy (~6.7 GB per chip) cached under <weight cache>/dp4tp8_*.
"""
import os
import pathlib
import time

import torch
from loguru import logger

import ttnn
from models.tt_dit.utils.tensor import prepare_for_fused_swiglu

DRAM = ttnn.DRAM_MEMORY_CONFIG
TP = 8
COLS = 4


def dp4_enabled() -> bool:
    return os.environ.get("LLAMA_BH_PREFILL_DP4", "0") == "1"


def dp4_min_len() -> int:
    return int(os.environ.get("LLAMA_BH_PREFILL_DP4_MIN", "2048"))


class DP4TP8Prefill:
    def __init__(self, model, args, mesh_device, state_dict, weight_cache_path):
        assert tuple(mesh_device.shape) == (
            8,
            4,
        ), f"DP4 x TP8 prefill needs the 8x4 mesh, got {tuple(mesh_device.shape)}"
        self.model = model
        self.args = args
        self.mesh = mesh_device
        self.n_layers = args.n_layers
        self.dim = args.dim
        self.head_dim = args.head_dim
        self.nq = args.n_heads // TP
        self.nkv = args.n_kv_heads // TP
        assert self.nkv == 1, "one kv head per chip expected (8 kv heads over 8 rows)"
        self.ff = args.hidden_dim // TP
        self.eps = args.norm_eps
        self.grid = mesh_device.compute_with_storage_grid_size()
        crs = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(self.grid.x - 1, self.grid.y - 1))]
        )
        self._ag_sems = [[ttnn.create_global_semaphore(mesh_device, crs, 0) for _ in range(2)] for _ in range(2)]
        self._ag_bars = [ttnn.create_global_semaphore(mesh_device, crs, 0) for _ in range(2)]
        self._ag_i = 0
        # LLAMA_BH_PREFILL_DP4_LAYOUT=hidden: residual hidden-sharded [M, 8192/8] per chip; QKV and FF1|FF3 run as the
        # fused, overlapping all-gather + matmul (all_gather_minimal_matmul_async, K-gather), WO / FF2 reduce-scatter N.
        self.layout = os.environ.get("LLAMA_BH_PREFILL_DP4_LAYOUT", "seq")
        self._agmm_sems = [[ttnn.create_global_semaphore(mesh_device, crs, 0) for _ in range(2)] for _ in range(4)]
        self._agmm_bars = [ttnn.create_global_semaphore(mesh_device, crs, 0) for _ in range(4)]
        self._agmm_i = 0
        self.links = int(os.environ.get("LLAMA_BH_PREFILL_DP4_LINKS", "2"))
        self.kb = int(os.environ.get("LLAMA_BH_PREFILL_DP4_KB") or "16")
        self.kb_ff13 = int(os.environ.get("LLAMA_BH_PREFILL_DP4_KB_FF13") or "32")
        # QKV matmul output in bf8 (LLAMA_BH_PREFILL_DP4_QKV8): heads/rotary read half the bytes and V needs no typecast
        # (rotary_embedding_llama takes a bf8 input in interleaved prefill). 32x16K TTFT 7.25 -> 7.18 s.
        # Default on since Nov 22: full MMLU (14042 questions, every prompt on DP4) with SDPA LoFi + QKV bf8 + WO a2a fp32
        # is 83.13% vs 82.99% for the old defaults (150 lost / 170 gained); the quick-set drop was noise. =0 turns it off.
        self.qkv8 = os.environ.get("LLAMA_BH_PREFILL_DP4_QKV8", "1") == "1"
        # all-gather workers per link (tests/test_ag_tune.py, 16K rows: 4 -> 79.8 GB/s, 8 -> 85.5 GB/s)
        self.ag_wpl = int(os.environ.get("LLAMA_BH_PREFILL_DP4_AG_WPL") or "8")
        # SDPA query in bf8 from this length (tests/test_dp4_sdpa_sweep.py: 16K 4.10 -> 3.75 ms, 4K no change); SDPA then
        # emits bf8, so the head concat needs no typecast before WO
        self.sdpa_q8_min = int(os.environ.get("LLAMA_BH_PREFILL_DP4_SDPA_Q8_MIN") or "0")
        # LLAMA_BH_PREFILL_DP4_RGMM: comma list of qkv / ff13 run as the fused row all-gather + matmul
        # (strided_all_gather_minimal_matmul_async dim=2) for prompts of at least RGMM_MIN tokens: each matmul core
        # row takes a 1/8 sub-band of every chip's rows and walks the chips in ring-arrival order, so the gather
        # (all_gather_async's mux transport, rows RGMM_GY..) hides under the matmul (grid.x x RGMM_GY). One column,
        # tests/test_row_gather_mm.py: QKV 16K 2.45 -> 2.13 ms, 32K 4.75 -> 4.19 ms; FF13 loses (the matmul gives
        # up 2 core rows: 16K 6.24 -> 6.77 ms), so it is not in the default list.
        self.rgmm = set(filter(None, (os.environ.get("LLAMA_BH_PREFILL_DP4_RGMM") or "").split(",")))
        self.rgmm_min = int(os.environ.get("LLAMA_BH_PREFILL_DP4_RGMM_MIN") or "16384")
        self.rgmm_gy = int(os.environ.get("LLAMA_BH_PREFILL_DP4_RGMM_GY") or "8")
        self._rg_sems = [[ttnn.create_global_semaphore(mesh_device, crs, 0) for _ in range(3)] for _ in range(2)]
        self.rgmm_wpl = int(os.environ.get("LLAMA_BH_PREFILL_DP4_RGMM_WPL") or "4")
        self._rg_i = 0

        def ckc(fid, fp32=False):
            return ttnn.init_device_compute_kernel_config(
                mesh_device.arch(), math_fidelity=fid, math_approx_mode=False, fp32_dest_acc_en=fp32, packer_l1_acc=True
            )

        # fp32 accumulation per matmul (LLAMA_BH_PREFILL_DP4_FP32, comma list of qkv / ff13): TP8 accumulates the full
        # K = 8192 on one chip, 4x the TP32 per-chip K, so bf16 accumulation loses more than it does in TP32.
        fp32 = set((os.environ.get("LLAMA_BH_PREFILL_DP4_FP32") or "qkv,ff13").split(","))
        # Opt-in precision change (LLAMA_BH_PREFILL_DP4_LOFI, comma list of qkv / wo / ff2): run those bf8-weight matmuls
        # at LoFi (2x the HiFi2 rate), the trade the WH config makes with bfp4 FF2 / attention weights.
        self.lofi = set(filter(None, (os.environ.get("LLAMA_BH_PREFILL_DP4_LOFI") or "").split(",")))
        self.ck_qkv = ckc(ttnn.MathFidelity.LoFi if "qkv" in self.lofi else ttnn.MathFidelity.HiFi2, fp32="qkv" in fp32)
        self.ck_ff13 = ckc(ttnn.MathFidelity.LoFi, fp32="ff13" in fp32)
        self.ck_norm = ckc(ttnn.MathFidelity.HiFi4, fp32=True)
        # SDPA: HiFi2 with fp32 accumulation like the TP32 prefill, but the approximate exp: with fp32 accumulation
        # the exact exp makes the op 2.6x slower (8 heads, causal, one chip: 4K 1.45 -> 0.57 ms, 16K 15.7 -> 5.8 ms)
        # for an op-level PCC vs torch of 0.99975 instead of 0.99995 (tests/test_dp4_sdpa_sweep.py).
        # LLAMA_BH_PREFILL_DP4_SDPA_FP32=0 drops fp32 accumulation; LLAMA_BH_PREFILL_DP4_SDPA_APPROX=0 the approx exp.
        # LLAMA_BH_PREFILL_DP4_SDPA_FID: LoFi by default since Nov 22 (-3% per 16K layer; full MMLU neutral, see QKV8).
        self.sdpa_fp32 = os.environ.get("LLAMA_BH_PREFILL_DP4_SDPA_FP32", "1") == "1"
        self.sdpa_approx = os.environ.get("LLAMA_BH_PREFILL_DP4_SDPA_APPROX", "1") == "1"
        self.ck_sdpa = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=getattr(ttnn.MathFidelity, os.environ.get("LLAMA_BH_PREFILL_DP4_SDPA_FID") or "LoFi"),
            math_approx_mode=self.sdpa_approx,
            fp32_dest_acc_en=self.sdpa_fp32,
            packer_l1_acc=True,
        )
        # Final-layer trim (LLAMA_BH_PREFILL_DP4_TRIM_LAST, default on from LLAMA_BH_PREFILL_DP4_TRIM_MIN = 8192 tokens):
        # the last layer's attention, WO, residual add, norm, FF13 and FF2 only matter for each prompt's last token
        # (every other row is discarded; the KV cache is written before attention), so they run on that one row.
        # Same logits (test_dp4tp8_pcc.py, pcc to 4 decimals). 32x16K TTFT 7.23 -> 7.14 s (3 seeds each). At 4K the
        # saving is ~1% and the closed-loop 32x4K line got worse (1.61 -> 1.70 s: later rounds arrive bunched), so 4K
        # keeps the full layer.
        self.trim_last = os.environ.get("LLAMA_BH_PREFILL_DP4_TRIM_LAST", "1") == "1"
        self.trim_min = int(os.environ.get("LLAMA_BH_PREFILL_DP4_TRIM_MIN") or 8192)
        # with the trim, the last layer's attention also runs for the last query row only (=0 keeps full attention)
        self.trim_attn = os.environ.get("LLAMA_BH_PREFILL_DP4_TRIM_ATTN", "1") == "1"
        self.ck_wo_row = ckc(ttnn.MathFidelity.LoFi if "wo" in self.lofi else ttnn.MathFidelity.HiFi2)
        # the WO-a2a matmul sums K = 8192 on one chip (MMRS sums eight K = 1024 partials in an fp32 hold):
        # LLAMA_BH_PREFILL_DP4_WO_A2A_FP32=1 accumulates it in fp32
        self.ck_wo_mm = ckc(
            ttnn.MathFidelity.LoFi if "wo" in self.lofi else ttnn.MathFidelity.HiFi2,
            fp32=os.environ.get("LLAMA_BH_PREFILL_DP4_WO_A2A_FP32", "1") == "1",
        )
        self.ck_ff2_row = ckc(ttnn.MathFidelity.LoFi if "ff2" in self.lofi else ttnn.MathFidelity.HiFi2)
        self._sel_ramps = {}
        self.rows_out = False  # True when the last forward returned last-token rows instead of the SP residual
        self.mmrs_ck_wo = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.LoFi if "wo" in self.lofi else ttnn.MathFidelity.HiFi2
        )
        self.mmrs_ck_ff2 = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.LoFi if "ff2" in self.lofi else ttnn.MathFidelity.HiFi2
        )
        rows_only = ttnn.ShardTensor2dMesh(
            mesh_device, dims=(0, None), mesh_shape=(8, 4)
        )  # mesh axis 0 <- tensor dim 0
        rep = ttnn.ReplicateTensorToMesh(mesh_device)

        def cache(name):
            return None if weight_cache_path is None else weight_cache_path / f"dp4tp8_{name}"

        self.fused_swiglu = os.environ.get("LLAMA_BH_PREFILL_DP4_SWIGLU", "1") == "1"
        self.norm_old = os.environ.get("LLAMA_BH_PREFILL_DP4_NORM_OLD", "0") == "1"
        # LLAMA_BH_PREFILL_DP4_NORM=rms: one ttnn.rms_norm (its streaming large-row kernel) + typecast to bf8 instead of
        # the 5-op ones-column norm. One chip (tests/test_dp4_norm_fast.py): 4K rows 0.35 -> 0.20 ms, 16K 0.80 -> 0.63 ms
        # per norm incl. the residual add, same PCC.
        self.norm_rms = os.environ.get("LLAMA_BH_PREFILL_DP4_NORM", "") == "rms"
        # LLAMA_BH_PREFILL_DP4_WO_A2A_MIN (default 8192, 0 = off): from this many tokens, WO runs without the
        # reduce-scatter. An all-to-all over the column hands chip r every head of ITS rows ([M/8, 8192], ~15 MB moved
        # at 16K instead of the ~124 MB of [M, 8192] partials) and chip r applies the full WO (a third weight copy of
        # WO, ~5.7 GB per chip). Same FLOPs and output layout; closer to torch than the bf8-transport MMRS (tests/test_wo_a2a.py: pcc
        # 1.0000 vs 0.9988 at 16K). Per 16K layer at the power cap 19.24-19.49 -> 19.01-19.12 ms; 4K slightly slower
        # (tests/test_op_energy.py layer variants). Served 32 x 16K (3 seeds): 7.057 s vs 7.107 s. Default on with fp32
        # accumulation (LLAMA_BH_PREFILL_DP4_WO_A2A_FP32): full MMLU neutral (see QKV8).
        self.wo_a2a_min = int(os.environ.get("LLAMA_BH_PREFILL_DP4_WO_A2A_MIN") or 8192)
        # LLAMA_BH_PREFILL_DP4_W2B4_MIN (default 2048, 0 = off): from this many tokens FF2 uses a bfp4 copy of W2 (~1.3 GB
        # per chip, cache dp4tp8_L*.w2b4). At the cap: FF2 MMRS 16K 3.70 -> 3.43 ms; 4K 0.97 -> 0.84 ms under the
        # in-model L1 budget (tests/test_mrs_energy.py; without that headroom the planner picks a slow 4K plan, which made
        # it look slower in tests/test_op_energy.py). 4 x 4K forward 0.376 -> 0.363 s. Full MMLU (every prompt on DP4,
        # forced from 512 tokens) 82.92% vs 82.99% control (245 lost / 236 gained). Served: 32 x 16K 6.798 -> 6.746 s,
        # 32 x 4K round-0 step spacing 0.388-0.392 -> 0.382 s.
        self.w2b4_min = int(os.environ.get("LLAMA_BH_PREFILL_DP4_W2B4_MIN") or 2048)
        # LLAMA_BH_PREFILL_DP4_ATTN_B4_MIN (0 = off): from this many tokens QKV and the WO-a2a matmul use bfp4 weight copies
        # (cache dp4tp8_L*.wqkvb4 / wo_fullb4, ~0.4 + 2.9 GB per chip). Alone at 16K: WO full matmul -30% energy
        # (wo_mm 1.38 -> 1.00 ms at the cap), QKV bfp4 ~-1% per layer.
        self.attn_b4_min = int(os.environ.get("LLAMA_BH_PREFILL_DP4_ATTN_B4_MIN") or 0)
        prefix = args.get_state_dict_prefix("", None)
        H, D, F = self.dim, self.head_dim, args.hidden_dim
        W13 = "w13sg2" if self.fused_swiglu else "w13"
        names = (
            ("wqkv", "wo", W13, "w2") + (("wo_full",) if self.wo_a2a_min else ()) + (("w2b4",) if self.w2b4_min else ())
        )
        names += (("wqkvb4",) + (("wo_fullb4",) if self.wo_a2a_min else ())) if self.attn_b4_min else ()

        def cache_hit(i, name):
            c = cache(f"L{i}.{name}")
            return c is not None and any(pathlib.Path(c).parent.glob(pathlib.Path(c).name + "_dtype_*.tensorbin"))

        missing = [(i, n) for i in range(self.n_layers) for n in names if not cache_hit(i, n)]
        if missing:
            # A warm TP32 weight cache gives the model a placeholder state dict (models/common/weight_cache.py
            # CachedStateDict: keys without real host tensors). Building from it writes zeros: load the real one.
            from models.common.weight_cache import CachedStateDict

            if state_dict is None or isinstance(state_dict, CachedStateDict):
                logger.info(
                    f"DP4xTP8: {len(missing)} weight cache files missing, loading the HF state dict to build them"
                )
                state_dict = args.load_state_dict()
        self.layers = []
        for i in range(self.n_layers):
            p = f"{prefix}layers.{i}."
            have = state_dict is not None and f"{p}attention.wq.weight" in state_dict
            g_attn = state_dict[f"{p}attention_norm.weight"].float() if have else None
            g_ff = state_dict[f"{p}ffn_norm.weight"].float() if have else None

            def build_qkv():
                wq, wk, wv = (state_dict[f"{p}attention.{n}.weight"].float() for n in ("wq", "wk", "wv"))
                per = []
                for r in range(TP):
                    q = wq[r * self.nq * D : (r + 1) * self.nq * D]
                    k = wk[r * D : (r + 1) * D]
                    v = wv[r * D : (r + 1) * D]
                    w = torch.cat([q, k, v], dim=0).t() * g_attn[:, None]  # [H, 1280], gamma folded on K
                    per.append(w.unsqueeze(0))
                return torch.cat(per, dim=0).unsqueeze(1)  # [8, 1, H, 1280]

            def build_wo():
                wo = state_dict[f"{p}attention.wo.weight"].float()  # [H out, H in]
                return torch.cat(
                    [wo[:, r * self.nq * D : (r + 1) * self.nq * D].t() for r in range(TP)], dim=0
                )  # [8*1024, H]

            def build_w13_plain():
                w1 = state_dict[f"{p}feed_forward.w1.weight"].float()
                w3 = state_dict[f"{p}feed_forward.w3.weight"].float()
                f = self.ff
                per = [
                    (
                        torch.cat([w1[r * f : (r + 1) * f], w3[r * f : (r + 1) * f]], dim=0).t() * g_ff[:, None]
                    ).unsqueeze(0)
                    for r in range(TP)
                ]
                return torch.cat(per, dim=0).unsqueeze(1)

            def build_w13():
                w1 = state_dict[f"{p}feed_forward.w1.weight"].float()  # [F, H]
                w3 = state_dict[f"{p}feed_forward.w3.weight"].float()
                f = self.ff
                per = []
                for r in range(TP):
                    gate_up = torch.cat([w1[r * f : (r + 1) * f], w3[r * f : (r + 1) * f]], dim=0).t() * g_ff[:, None]
                    # [gate | up] -> interleaved tile pairs for minimal_matmul(fuse_swiglu=True): silu(gate) * up
                    per.append(prepare_for_fused_swiglu(gate_up, ndev=1, gate_is_first=True).unsqueeze(0))
                return torch.cat(per, dim=0).unsqueeze(1)  # [8, 1, H, 2f]

            def build_w2():
                w2 = state_dict[f"{p}feed_forward.w2.weight"].float()  # [H, F]
                return torch.cat([w2[:, r * self.ff : (r + 1) * self.ff].t() for r in range(TP)], dim=0)  # [8*f, H]

            def load(name, builder, shape, dtype, mapper=None):
                c = cache(f"L{i}.{name}")
                hit = c is not None and any(pathlib.Path(c).parent.glob(pathlib.Path(c).name + "_dtype_*.tensorbin"))
                # a cache hit never reads the torch tensor: skip building it
                if not hit and not have:
                    raise RuntimeError(f"DP4xTP8: no cache file and no weights for layer {i} {name}")
                torch_w = torch.empty(0) if hit else builder().to(torch.bfloat16)
                return ttnn.as_tensor(
                    torch_w,
                    dtype=dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh_device,
                    memory_config=DRAM,
                    mesh_mapper=mapper or rows_only,
                    cache_file_name=cache(f"L{i}.{name}"),
                )

            if i == 0 and os.environ.get("LLAMA_BH_PREFILL_DP4_SWIGLU_CHECK", "0") == "1":
                self._w13_plain0 = load("w13", build_w13_plain, (TP, 1, H, 2 * self.ff), ttnn.bfloat4_b)
            self.layers.append(
                {
                    "wqkv": load("wqkv", build_qkv, (TP, 1, H, (self.nq + 2) * D), ttnn.bfloat8_b),
                    "wo": load("wo", build_wo, (TP * self.nq * D, H), ttnn.bfloat8_b),
                    "w13": load("w13sg2", build_w13, (TP, 1, H, 2 * self.ff), ttnn.bfloat4_b)
                    if self.fused_swiglu
                    else load("w13", build_w13_plain, (TP, 1, H, 2 * self.ff), ttnn.bfloat4_b),
                    "w2": load("w2", build_w2, (TP * self.ff, H), ttnn.bfloat8_b),
                }
            )
            if self.w2b4_min:
                self.layers[-1]["w2b4"] = load("w2b4", build_w2, (TP * self.ff, H), ttnn.bfloat4_b)
            if self.attn_b4_min:
                self.layers[-1]["wqkvb4"] = load("wqkvb4", build_qkv, (TP, 1, H, (self.nq + 2) * D), ttnn.bfloat4_b)
                if self.wo_a2a_min:
                    self.layers[-1]["wo_fullb4"] = load(
                        "wo_fullb4", lambda: build_wo().reshape(1, 1, H, H), (1, 1, H, H), ttnn.bfloat4_b, mapper=rep
                    )
            if self.wo_a2a_min:
                # full WO [H in, H out] on every chip: build_wo stacks the per-row shards, i.e. it is wo.t() whole
                self.layers[-1]["wo_full"] = load(
                    "wo_full", lambda: build_wo().reshape(1, 1, H, H), (1, 1, H, H), ttnn.bfloat8_b, mapper=rep
                )
        self.ones_col = ttnn.from_torch(
            torch.full((1, 1, H, 32 if self.norm_old else 1), 1.0 / H),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=DRAM,
            mesh_mapper=rep,
        )
        tm = torch.zeros(1, 1, 32, 32)
        tm[..., torch.arange(0, 32, 2), torch.arange(1, 32, 2)] = 1
        tm[..., torch.arange(1, 32, 2), torch.arange(0, 32, 2)] = -1
        self.trans = ttnn.from_torch(
            tm, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=rep
        )
        self._rope = {}
        self._prof = None
        self._traces = {}
        self._seen = set()
        logger.info(
            f"DP4xTP8 prefill ready: {self.n_layers} layers, links={self.links}, fp32 acc {sorted(fp32)}, sdpa fp32 {self.sdpa_fp32} approx {self.sdpa_approx}, lofi {sorted(self.lofi)}, layout {self.layout}"
        )

    # ------------------------------------------------------------------ pieces
    def _ag(self, x, dim=2):
        i = self._ag_i
        self._ag_i ^= 1
        return ttnn.experimental.all_gather_async(
            input_tensor=x,
            persistent_output_buffer=None,
            dim=dim,
            multi_device_global_semaphore=self._ag_sems[i],
            num_links=self.links,
            barrier_semaphore=self._ag_bars[i],
            memory_config=DRAM,
            topology=ttnn.Topology.Ring,
            cluster_axis=0,
            num_workers_per_link=self.ag_wpl,
        )

    def _norm(self, x):
        """x [1,1,R,8192] bf16 -> x / rms(x) in bf8 (gamma lives in the next weight)."""
        if self.norm_rms:
            try:
                # bf8 straight out of the norm kernel (rms_norm dtype=, tt-metal change of Nov 21): no typecast pass
                return ttnn.rms_norm(
                    x, epsilon=self.eps, memory_config=DRAM, compute_kernel_config=self.ck_norm, dtype=ttnn.bfloat8_b
                )
            except TypeError:
                n = ttnn.rms_norm(x, epsilon=self.eps, memory_config=DRAM, compute_kernel_config=self.ck_norm)
                out = ttnn.typecast(n, ttnn.bfloat8_b)
                ttnn.deallocate(n)
                return out
        if self.norm_old:
            sq = ttnn.mul(x, x, dtype=ttnn.bfloat16, memory_config=DRAM)
            ms = ttnn.matmul(
                sq, self.ones_col, compute_kernel_config=self.ck_norm, dtype=ttnn.float32, memory_config=DRAM
            )
            ttnn.deallocate(sq)
            r = ttnn.rsqrt(ttnn.add(ms[:, :, :, :1], self.eps, memory_config=DRAM), memory_config=DRAM)
            ttnn.deallocate(ms)
            r16 = ttnn.typecast(r, ttnn.bfloat16)
            ttnn.deallocate(r)
            out = ttnn.mul(x, r16, dtype=ttnn.bfloat8_b, memory_config=DRAM)
            ttnn.deallocate(r16)
            return out
        sq = ttnn.mul(x, x, dtype=ttnn.bfloat16, memory_config=DRAM)
        ms = ttnn.matmul(sq, self.ones_col, compute_kernel_config=self.ck_norm, dtype=ttnn.bfloat16, memory_config=DRAM)
        ttnn.deallocate(sq)
        ms_eps = ttnn.add(ms, self.eps, memory_config=DRAM)
        ttnn.deallocate(ms)
        r = ttnn.rsqrt(ms_eps, memory_config=DRAM)  # [R, 1]
        ttnn.deallocate(ms_eps)
        out = ttnn.mul(x, r, dtype=ttnn.bfloat8_b, memory_config=DRAM)
        ttnn.deallocate(r)
        return out

    def _mm(self, a, w, ck, dtype, swiglu=False, kb=None):
        # fp32 accumulation halves the dest register: 4-tile subblocks instead of 8
        sbw = 4 if getattr(ck, "fp32_dest_acc_en", False) else 8
        # K block 16 (LLAMA_BH_PREFILL_DP4_KB): one chip, tests/test_dp4_mm_acc.py: FF13 16K 4.70 -> 4.51 ms (fp32 acc),
        # QKV 16K 0.86 -> 0.78 ms, same PCC
        cfg = ttnn.MinimalMatmulConfig(
            M_block_size=8,
            K_block_size=kb or self.kb,
            N_block_size=8,
            subblock_h=1,
            subblock_w=sbw,
            compute_with_storage_grid_size=self.grid,
        )
        return ttnn.experimental.minimal_matmul(
            input_tensor=a,
            weight_tensor=w,
            config=cfg,
            compute_kernel_config=ck,
            memory_config=DRAM,
            dtype=dtype,
            fuse_swiglu=swiglu,
        )

    def _rgmm(self, x, w, ck, dtype, M, swiglu=False):
        """Fused row all-gather (column axis) + matmul: x [1,1,M/8,8192] per chip -> x_all @ w, [1,1,M,N]."""
        sbw = 4 if getattr(ck, "fp32_dest_acc_en", False) else 8
        sub = M // 32 // (TP * TP)  # row tiles per core row per chip slice; M blocks must tile it
        mb = next(m for m in (8, 4, 2, 1) if sub % m == 0)
        cfg = ttnn.MinimalMatmulConfig(
            M_block_size=mb,
            K_block_size=8,
            N_block_size=8,
            subblock_h=1,
            subblock_w=sbw,
            compute_with_storage_grid_size=ttnn.CoreCoord(self.grid.x, self.rgmm_gy),
        )
        i = self._rg_i
        self._rg_i ^= 1
        out = ttnn.experimental.strided_all_gather_minimal_matmul_async(
            x,
            w,
            None,
            2,
            self._rg_sems[i],
            ttnn.CoreCoord(0, self.rgmm_gy),
            num_links=self.links,
            num_workers_per_link=self.rgmm_wpl,
            memory_config_ag=DRAM,
            topology=ttnn.Topology.Ring,
            cluster_axis=0,
            config=cfg,
            memory_config_mm=DRAM,
            compute_kernel_config=ck,
            fuse_swiglu=swiglu,
        )
        ttnn.deallocate(out[0])
        return ttnn.typecast(out[1], dtype) if out[1].dtype != dtype else out[1]

    def _mmrs(self, a, w, ck, scatter_dim=-2, l1_headroom=None):
        kw = {} if l1_headroom is None else {"l1_headroom": l1_headroom}
        out = ttnn.experimental.matmul_reduce_scatter_prefill(
            a,
            w,
            cluster_axis=0,
            scatter_dim=scatter_dim,
            topology=ttnn.Topology.Ring,
            num_links=self.links,
            compute_kernel_config=ck,
            memory_config=DRAM,
            **kw,
        )
        # LLAMA_BH_PREFILL_DP4_MRS_PLAN=1 logs each shape's plan once. In the model the live L1 tensors (decode CCL
        # buffers, semaphores: the top ~180 KB of L1) cap the plan footprint at ~1.04-1.08 MB; on an empty device the
        # planner picks 1.25-1.37 MB plans, e.g. FF2 16K 2.74 ms vs ~3.8-4.1 ms in the model.
        if os.environ.get("LLAMA_BH_PREFILL_DP4_MRS_PLAN") == "1":
            key = (tuple(a.shape), tuple(w.shape), l1_headroom)
            if key not in getattr(self, "_mrs_logged", set()):
                self._mrs_logged = getattr(self, "_mrs_logged", set()) | {key}
                plan = ttnn._ttnn.operations.experimental.ccl_experimental._matmul_reduce_scatter_prefill_last_plan()
                logger.info(f"DP4xTP8 MRS plan a={key[0]} w={key[1]} hr={l1_headroom}: {dict(plan)}")
        return out

    def _rope_mats(self, M):
        if M not in self._rope:
            cos_full, sin_full = self.model.get_or_create_prefill_rot_mats()
            self._rope[M] = (
                ttnn.slice(cos_full, [0, 0, 0, 0], [1, 1, M, self.head_dim]),
                ttnn.slice(sin_full, [0, 0, 0, 0], [1, 1, M, self.head_dim]),
            )
        return self._rope[M]

    def _rotate(self, x, cos, sin, dtype):
        """RoPE; dtype=bfloat8_b packs the rotated q / k straight to bf8 (rotary_embedding_llama dtype=, tt-metal change
        of Nov 21), saving the typecast pass, with a typecast fallback on an older build."""
        if dtype is None:
            return ttnn.experimental.rotary_embedding_llama(x, cos, sin, self.trans, is_decode_mode=False)
        try:
            return ttnn.experimental.rotary_embedding_llama(x, cos, sin, self.trans, is_decode_mode=False, dtype=dtype)
        except TypeError:
            y = ttnn.experimental.rotary_embedding_llama(x, cos, sin, self.trans, is_decode_mode=False)
            z = ttnn.typecast(y, dtype)
            ttnn.deallocate(y)
            return z

    SDPA_CHUNKS = ((256, 512), (256, 256), (128, 256), (128, 128))

    def _sdpa(self, q, k, v, M):
        """Causal SDPA with the largest (q, k) chunks whose circular buffers fit next to the model's resident L1
        buffers; the choice is cached per length (a too-large config fails at program build, before running)."""
        if not hasattr(self, "_sdpa_ok"):
            self._sdpa_ok = {}
        # k chunk 1024 from 8K tokens (bf8 query, one chip, tests/test_dp4_sdpa_sweep.py: 8K 1.28 -> 1.25 ms, 16K
        # 3.70 -> 3.61 ms; 4K is best at 512)
        order = (((256, 1024),) if M >= 8192 else ()) + self.SDPA_CHUNKS
        cands = [self._sdpa_ok[M]] if M in self._sdpa_ok else [c for c in order if M % c[0] == 0 and M % c[1] == 0]
        err = None
        for qc, kc in cands:
            cfg = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=self.grid,
                q_chunk_size=qc,
                k_chunk_size=kc,
                exp_approx_mode=self.sdpa_approx,
            )
            try:
                o = ttnn.transformer.scaled_dot_product_attention(
                    q, k, v, is_causal=True, program_config=cfg, compute_kernel_config=self.ck_sdpa, memory_config=DRAM
                )
            except RuntimeError as e:
                err = e
                logger.warning(f"DP4xTP8 SDPA q{qc} k{kc} at M={M} does not fit, trying smaller chunks")
                continue
            if M not in self._sdpa_ok:
                logger.info(f"DP4xTP8 SDPA M={M}: q{qc} k{kc}")
            self._sdpa_ok[M] = (qc, kc)
            return o
        raise err

    # ------------------------------------------------------------------ inputs
    def _host_tokens(self, tokens_cols, M):
        """tokens_cols: [4, M] int (row c = prompt of column c) -> host tensor, chip (r, c) gets rows r*R..(r+1)*R of
        ALL four prompts (it embeds them with its column's hidden quarter; see _embed)."""
        R = M // TP
        t = tokens_cols.reshape(COLS, TP, R).permute(1, 0, 2).reshape(TP, 1, 1, COLS * R)  # [8, 1, 1, 4R]
        return ttnn.from_torch(
            t.to(torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh, dims=(0, None), mesh_shape=(8, 4)),
        )

    def _embed(self, tt_tok, M):
        """device tokens (from _host_tokens) -> [1,1,M/8,8192] bf16 residual, SP layout (prompt c on column c)."""
        R = M // TP
        tok2 = ttnn.reshape(tt_tok, ttnn.Shape((1, COLS * R)))
        e = ttnn.embedding(
            tok2, self.model.embd.weights, layout=ttnn.TILE_LAYOUT, memory_config=DRAM, dtype=ttnn.bfloat16
        )
        e = ttnn.reshape(e, ttnn.Shape((1, COLS, R, self.dim // COLS)))
        g = ttnn.all_gather(e, 3, cluster_axis=1, memory_config=DRAM)  # [1, 4, R, 8192]
        ttnn.deallocate(e)
        x = ttnn.mesh_partition(g, dim=1, cluster_axis=1, memory_config=DRAM)  # [1, 1, R, 8192]: prompt c
        ttnn.deallocate(g)
        return x

    def _page_table(self, page_tables_cols, device=True):
        """page_tables_cols: [4, nb] int32 (row c = the user on column c; -1 = no user / skip writes)."""
        nb = page_tables_cols.shape[1]
        nbp = ((nb + 7) // 8) * 8
        pt = torch.full((COLS, nbp), -1, dtype=torch.int32)
        pt[:, :nb] = page_tables_cols
        return ttnn.from_torch(
            pt,
            device=self.mesh if device else None,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh, dims=(None, 0), mesh_shape=(8, 4)),
        )

    # ------------------------------------------------------------ hidden-sharded layout
    # Tuned all_gather_minimal_matmul_async configs, (M, N, swiglu) -> (M_block, K_block, N_block, sb_h, sb_w,
    # transpose, buffers per channel); from tests/test_agmm_sweep.py on tt-galaxy-43. Missing shapes use tt_dit's rules.
    AGMM_TABLE = {}

    def _agmm_cfg(self, M, N, swiglu, fp32):
        full = self.grid
        key = (M, N, swiglu)
        if key in self.AGMM_TABLE:
            mb, kb, nb, sh, sw, transpose, bufs = self.AGMM_TABLE[key]
            if fp32 and sh * sw > 4:
                sh, sw = 1, 4
            grid = ttnn.CoreCoord(full.x, full.y - 1) if transpose else ttnn.CoreCoord(full.x - 1, full.y)
            workers = -(-(grid.x if transpose else grid.y) // self.links)
            cfg = ttnn.MinimalMatmulConfig(
                M_block_size=mb,
                K_block_size=kb,
                N_block_size=nb,
                subblock_h=sh,
                subblock_w=sw,
                compute_with_storage_grid_size=grid,
            )
            return cfg, transpose, workers, bufs
        from models.tt_dit.utils.matmul import get_agmm_config

        _, cfg, workers = get_agmm_config(
            M,
            self.dim,
            N,
            full_grid=full,
            cluster_size=TP,
            num_links=self.links,
            force_transpose=True,
            fuse_swiglu=swiglu,
        )
        if fp32 and cfg.subblock_h * cfg.subblock_w > 4:
            cfg = ttnn.MinimalMatmulConfig(
                M_block_size=cfg.M_block_size,
                K_block_size=cfg.K_block_size,
                N_block_size=cfg.N_block_size,
                subblock_h=1,
                subblock_w=4,
                compute_with_storage_grid_size=cfg.compute_with_storage_grid_size,
            )
        return cfg, True, workers, 24

    def _agmm(self, x, w, N, ck, swiglu, dtype, M):
        cfg, transpose, workers, bufs = self._agmm_cfg(M, N, swiglu, getattr(ck, "fp32_dest_acc_en", False))
        i = self._agmm_i
        self._agmm_i = (i + 1) % 4
        out = ttnn.experimental.all_gather_minimal_matmul_async(
            input_tensor=x,
            weight_tensor=w,
            config=cfg,
            compute_kernel_config=ck,
            multi_device_global_semaphore=self._agmm_sems[i],
            barrier_semaphore=self._agmm_bars[i],
            num_links=self.links,
            topology=ttnn.Topology.Ring,
            cluster_axis=0,
            memory_config=DRAM,
            dtype=dtype,
            force_transpose=transpose,
            num_workers_per_link=workers,
            num_buffers_per_channel=bufs,
            fuse_swiglu=swiglu,
        )
        for o in out[:-1]:
            ttnn.deallocate(o)
        return out[-1]

    def _norm_dist(self, x):
        """x [1,1,M,1024] bf16 hidden-sharded over the column -> x / rms(x) in bf8 (gamma lives in the next weight)."""
        st = ttnn.rms_norm_pre_all_gather(x, dtype=ttnn.bfloat16)
        g = self._ag(st, dim=3)
        ttnn.deallocate(st)
        o = ttnn.rms_norm_post_all_gather(x, g, epsilon=self.eps, memory_config=DRAM, dtype=ttnn.bfloat8_b)
        ttnn.deallocate(g)
        return o

    def _seq_to_hidden(self, x):
        """[1,1,M/8,8192] (rows) -> [1,1,M,1024] (hidden columns), one all-to-all over the column."""
        y = ttnn.experimental.all_to_all_async_generic(
            x, 2, 3, num_links=self.links, memory_config=DRAM, topology=ttnn.Topology.Ring, cluster_axis=0
        )
        return y

    def layer_hidden(self, i, x, M, cos, sin, page_table, kv):
        L = self.layers[i]
        mark = self._mark
        mark("start")
        h = self._norm_dist(x)
        mark("norm1")
        qkv = self._agmm(h, L["wqkv"], (self.nq + 2 * self.nkv) * self.head_dim, self.ck_qkv, False, ttnn.bfloat16, M)
        ttnn.deallocate(h)
        mark("qkv_agmm")
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv, num_heads=self.nq, num_kv_heads=self.nkv, transpose_k_heads=False, memory_config=DRAM
        )
        ttnn.deallocate(qkv)
        q2 = self._rotate(q, cos, sin, ttnn.bfloat8_b if M >= self.sdpa_q8_min else None)
        k8 = self._rotate(k, cos, sin, ttnn.bfloat8_b)
        ttnn.deallocate(q), ttnn.deallocate(k)
        if v.dtype == ttnn.bfloat8_b:
            v8 = v
        else:
            v8 = ttnn.typecast(v, ttnn.bfloat8_b)
            ttnn.deallocate(v)
        mark("heads_rope")
        if kv is not None:
            ttnn.experimental.paged_fill_cache(kv[0], k8, page_table, batch_idx=0)
            ttnn.experimental.paged_fill_cache(kv[1], v8, page_table, batch_idx=0)
        mark("kv_fill")
        o = self._sdpa(q2, k8, v8, M)
        mark("sdpa")
        ttnn.deallocate(q2), ttnn.deallocate(k8), ttnn.deallocate(v8)
        a = ttnn.experimental.nlp_concat_heads(o, memory_config=DRAM)
        ttnn.deallocate(o)
        if a.dtype == ttnn.bfloat8_b:  # SDPA emits its query's dtype: bf8 under SDPA_Q8_MIN
            a8 = a
        else:
            a8 = ttnn.typecast(a, ttnn.bfloat8_b)
            ttnn.deallocate(a)
        mark("concat")
        r1 = self._mmrs(a8, L["wo"], self.mmrs_ck_wo, scatter_dim=-1)
        ttnn.deallocate(a8)
        mark("wo_mmrs")
        x1 = ttnn.add(x, r1, dtype=ttnn.bfloat16, memory_config=DRAM)
        ttnn.deallocate(r1), ttnn.deallocate(x)
        h = self._norm_dist(x1)
        mark("add_norm2")
        f = self._agmm(h, L["w13"], 2 * self.ff, self.ck_ff13, True, ttnn.bfloat8_b, M)
        ttnn.deallocate(h)
        mark("ff13_agmm")
        r2 = self._mmrs(f, L["w2"], self.mmrs_ck_ff2, scatter_dim=-1)
        ttnn.deallocate(f)
        mark("ff2_mmrs")
        x2 = ttnn.add(x1, r2, dtype=ttnn.bfloat16, memory_config=DRAM)
        ttnn.deallocate(r2), ttnn.deallocate(x1)
        mark("add2")
        return x2

    # ------------------------------------------------------------------ forward
    def _mark(self, name):
        if self._prof is not None:
            ttnn.synchronize_device(self.mesh)
            self._prof.append((name, time.perf_counter()))

    def layer(self, i, x, M, cos, sin, page_table, kv, last_sel=None):
        L = self.layers[i]
        mark = self._mark
        mark("start")
        h = self._norm(x)
        mark("norm1")
        if "qkv" in self.rgmm and M >= self.rgmm_min and (M // 32 // TP) % TP == 0:
            qkv = self._rgmm(h, L["wqkv"], self.ck_qkv, ttnn.bfloat16, M)
            ttnn.deallocate(h)
            mark("ag1")
        else:
            hg = self._ag(h)
            ttnn.deallocate(h)
            mark("ag1")
            wq = L["wqkvb4"] if self.attn_b4_min and M >= self.attn_b4_min else L["wqkv"]
            qkv = self._mm(hg, wq, self.ck_qkv, ttnn.bfloat8_b if self.qkv8 else ttnn.bfloat16)
            ttnn.deallocate(hg)
        mark("qkv")
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv, num_heads=self.nq, num_kv_heads=self.nkv, transpose_k_heads=False, memory_config=DRAM
        )
        ttnn.deallocate(qkv)
        q2 = self._rotate(q, cos, sin, ttnn.bfloat8_b if M >= self.sdpa_q8_min else None)
        k8 = self._rotate(k, cos, sin, ttnn.bfloat8_b)
        ttnn.deallocate(q), ttnn.deallocate(k)
        if v.dtype == ttnn.bfloat8_b:
            v8 = v
        else:
            v8 = ttnn.typecast(v, ttnn.bfloat8_b)
            ttnn.deallocate(v)
        mark("heads_rope")
        if kv is not None:
            ttnn.experimental.paged_fill_cache(kv[0], k8, page_table, batch_idx=0)
            ttnn.experimental.paged_fill_cache(kv[1], v8, page_table, batch_idx=0)
        mark("kv_fill")
        if last_sel is not None and self.trim_attn:
            ar = self._last_attn(q2, k8, v8, last_sel)
            ttnn.deallocate(q2), ttnn.deallocate(k8), ttnn.deallocate(v8)
            return self._tail_rows(L, x, None, last_sel, ar=ar)
        o = self._sdpa(q2, k8, v8, M)
        mark("sdpa")
        ttnn.deallocate(q2), ttnn.deallocate(k8), ttnn.deallocate(v8)
        a = ttnn.experimental.nlp_concat_heads(o, memory_config=DRAM)
        ttnn.deallocate(o)
        if a.dtype == ttnn.bfloat8_b:  # SDPA emits its query's dtype: bf8 under SDPA_Q8_MIN
            a8 = a
        else:
            a8 = ttnn.typecast(a, ttnn.bfloat8_b)
            ttnn.deallocate(a)
        mark("concat")
        if last_sel is not None:
            return self._tail_rows(L, x, a8, last_sel)
        if self.wo_a2a_min and M >= self.wo_a2a_min:
            g = ttnn.experimental.all_to_all_async_generic(
                a8, 3, 2, num_links=self.links, memory_config=DRAM, topology=ttnn.Topology.Ring, cluster_axis=0
            )  # [1,1,M,1024] heads of chip r -> [1,1,M/8,8192]: all heads of chip r's rows
            ttnn.deallocate(a8)
            wof = L["wo_fullb4"] if self.attn_b4_min and M >= self.attn_b4_min else L["wo_full"]
            r1 = self._mm(g, wof, self.ck_wo_mm, ttnn.bfloat8_b)
            ttnn.deallocate(g)
        else:
            r1 = self._mmrs(a8, L["wo"], self.mmrs_ck_wo)
            ttnn.deallocate(a8)
        mark("wo_mmrs")
        x1 = ttnn.add(x, r1, dtype=ttnn.bfloat16, memory_config=DRAM)
        ttnn.deallocate(r1)
        ttnn.deallocate(x)
        h = self._norm(x1)
        mark("add_norm2")
        if "ff13" in self.rgmm and self.fused_swiglu and M >= self.rgmm_min and (M // 32 // TP) % TP == 0:
            f = self._rgmm(h, L["w13"], self.ck_ff13, ttnn.bfloat8_b, M, swiglu=True)
            ttnn.deallocate(h)
            mark("ag2")
            hg = None
        else:
            hg = self._ag(h)
            ttnn.deallocate(h)
            mark("ag2")
        if hg is None:
            pass
        elif self.fused_swiglu:
            # K block 32 for FF13 (tests/test_dp4_mm_acc.py, 16K bf16 acc: 4.03 -> 3.98 ms)
            try:
                f = self._mm(
                    hg, L["w13"], self.ck_ff13, ttnn.bfloat8_b, swiglu=True, kb=self.kb_ff13
                )  # silu(x W1) * (x W3)
            except RuntimeError:
                logger.warning(f"DP4xTP8 FF13 K block {self.kb_ff13} does not fit at M={M}; using {self.kb}")
                self.kb_ff13 = self.kb
                f = self._mm(hg, L["w13"], self.ck_ff13, ttnn.bfloat8_b, swiglu=True, kb=self.kb_ff13)
            if i == 0 and getattr(self, "_w13_plain0", None) is not None:
                f13 = self._mm(hg, self._w13_plain0, self.ck_ff13, ttnn.bfloat8_b)
                f1, f3 = ttnn.split(f13, self.ff, dim=-1)
                fr = ttnn.mul(
                    f1, f3, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], dtype=ttnn.bfloat8_b, memory_config=DRAM
                )
                for d in (0, 5, 31):
                    a = ttnn.to_torch(ttnn.get_device_tensors(f)[d]).float().flatten()
                    b = ttnn.to_torch(ttnn.get_device_tensors(fr)[d]).float().flatten()
                    pc = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
                    hgt = ttnn.to_torch(ttnn.get_device_tensors(hg)[d]).float()
                    st = (
                        lambda t: f"mean {t.mean():.4f} std {t.std():.4f} nan {int(torch.isnan(t).sum())} inf {int(torch.isinf(t).sum())}"
                    )
                    logger.info(
                        f"DP4xTP8 SWIGLU_CHECK dev {d}: pcc {pc:.5f} | fused {st(a)} | split {st(b)} | input {st(hgt)}"
                    )
                for z in (f13, f1, f3, fr):
                    ttnn.deallocate(z)
        else:
            f13 = self._mm(hg, L["w13"], self.ck_ff13, ttnn.bfloat8_b)
            f1, f3 = ttnn.split(f13, self.ff, dim=-1)
            ttnn.deallocate(f13)
            f = ttnn.mul(
                f1, f3, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], dtype=ttnn.bfloat8_b, memory_config=DRAM
            )
            ttnn.deallocate(f1), ttnn.deallocate(f3)
        if hg is not None:
            ttnn.deallocate(hg)
        mark("ff13_swiglu")
        r2 = self._mmrs(f, L["w2b4"] if self.w2b4_min and M >= self.w2b4_min else L["w2"], self.mmrs_ck_ff2)
        ttnn.deallocate(f)
        mark("ff2_mmrs")
        x2 = ttnn.add(x1, r2, dtype=ttnn.bfloat16, memory_config=DRAM)
        ttnn.deallocate(r2), ttnn.deallocate(x1)
        mark("add2")
        return x2

    def _last_sel(self, M, last_idx_cols):
        """One-hot selectors for each column's last token, built on device: (sel_seq [1,1,32,M], the token's position on
        every chip of column c; sel_res [1,1,32,M/8], non-zero only on the chip whose residual shard holds it). A cached
        fp32 position ramp is compared with the uploaded indices (a [32, M] host one-hot would be a 33 MB upload at 16K).
        All 32 rows select the same token."""
        R = M // TP
        # LLAMA_BH_PREFILL_DP4_TRIM_KEEP_RAMPS=1 caches the ramps per length; by default they are rebuilt and freed each
        # forward (~4 MB upload at 16K), so nothing stays resident in DRAM between steps
        keep = os.environ.get("LLAMA_BH_PREFILL_DP4_TRIM_KEEP_RAMPS") == "1"
        if M not in self._sel_ramps:
            seq = torch.arange(M, dtype=torch.float32).expand(1, 1, 32, M).contiguous()
            res = torch.stack([torch.arange(r * R, (r + 1) * R, dtype=torch.float32) for r in range(TP)])
            res = res[:, None, None, :].expand(TP, 1, 32, R).contiguous()
            self._sel_ramps[M] = (
                ttnn.from_torch(
                    seq,
                    device=self.mesh,
                    dtype=ttnn.float32,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=DRAM,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
                ),
                ttnn.from_torch(
                    res,
                    device=self.mesh,
                    dtype=ttnn.float32,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=DRAM,
                    mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh, dims=(0, None), mesh_shape=(8, 4)),
                ),
            )
        seq_t, res_t = self._sel_ramps[M]
        li = torch.full((1, COLS, 32, 1), -1.0)
        for c, v in last_idx_cols.items():
            li[0, c, :, 0] = float(v)
        li_t = ttnn.from_torch(
            li,
            device=self.mesh,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=DRAM,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh, dims=(None, 1), mesh_shape=(8, 4)),
        )
        out = []
        for ramp in (seq_t, res_t):
            e = ttnn.eq(ramp, li_t, memory_config=DRAM)
            out.append(ttnn.typecast(e, ttnn.bfloat16))
            ttnn.deallocate(e)
        # additive attention mask for the token's query row: -1e9 on positions after it (prompt padding)
        g = ttnn.gt(seq_t, li_t, memory_config=DRAM)
        gb = ttnn.typecast(g, ttnn.bfloat16)
        ttnn.deallocate(g)
        out.append(ttnn.multiply(gb, -1e9, memory_config=DRAM))
        ttnn.deallocate(gb)
        ttnn.deallocate(li_t)
        if not keep:
            for t in self._sel_ramps.pop(M):
                ttnn.deallocate(t)
        return tuple(out)

    def _col_sum(self, t):
        """[1,1,32,W] per chip -> the sum over the 8 chips of the mesh column, on every chip of it."""
        g = ttnn.all_gather(t, 2, cluster_axis=0, memory_config=DRAM)  # [1,1,256,W]
        ttnn.deallocate(t)
        g = ttnn.reshape(g, ttnn.Shape((1, TP, 32, int(g.shape[-1]))))
        s = ttnn.sum(g, dim=1, keepdim=True)
        ttnn.deallocate(g)
        return s

    def _last_attn(self, q, k, v, sel):
        """Attention for each prompt's last token only: its query row (selected after rope) against all keys, padding
        masked. q [1,8,M,128], k / v [1,1,M,128] -> [1,1,32,1024] (all rows the token's attention output)."""
        sel_seq, _, mask = sel
        qc = ttnn.experimental.nlp_concat_heads(q, memory_config=DRAM)  # [1,1,M,1024]
        qr = ttnn.matmul(sel_seq, qc, compute_kernel_config=self.ck_norm, dtype=ttnn.bfloat16, memory_config=DRAM)
        ttnn.deallocate(qc)
        rm = ttnn.to_layout(qr, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(qr)
        qh = ttnn.permute(ttnn.reshape(rm, ttnn.Shape((1, 32, self.nq, self.head_dim))), (0, 2, 1, 3))
        ttnn.deallocate(rm)
        qt = ttnn.to_layout(qh, ttnn.TILE_LAYOUT)
        ttnn.deallocate(qh)
        cfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.grid,
            q_chunk_size=32,
            k_chunk_size=512,
            exp_approx_mode=self.sdpa_approx,
        )
        o = ttnn.transformer.scaled_dot_product_attention(
            qt,
            k,
            v,
            is_causal=False,
            attn_mask=mask,
            program_config=cfg,
            compute_kernel_config=self.ck_sdpa,
            memory_config=DRAM,
        )
        ttnn.deallocate(qt)
        a = ttnn.experimental.nlp_concat_heads(o, memory_config=DRAM)
        ttnn.deallocate(o)
        return a

    def _tail_rows(self, L, x, a8, sel, ar=None):
        """The last layer after attention, on each prompt's last token only. x: SP residual [1,1,M/8,8192];
        a8: attention output [1,1,M,1024] (this chip's heads, every row). Returns [1,1,32,8192] per chip: the token's
        final residual on every chip of its column (all 32 rows equal)."""
        sel_seq, sel_res = sel[0], sel[1]
        if ar is None:
            ar = ttnn.matmul(sel_seq, a8, compute_kernel_config=self.ck_norm, dtype=ttnn.bfloat16, memory_config=DRAM)
            ttnn.deallocate(a8)
        p = ttnn.matmul(ar, L["wo"], compute_kernel_config=self.ck_wo_row, dtype=ttnn.bfloat16, memory_config=DRAM)
        ttnn.deallocate(ar)
        xr = ttnn.matmul(sel_res, x, compute_kernel_config=self.ck_norm, dtype=ttnn.bfloat16, memory_config=DRAM)
        ttnn.deallocate(x)
        s = ttnn.add(p, xr, dtype=ttnn.bfloat16, memory_config=DRAM)  # residual row only on its owner chip
        ttnn.deallocate(p), ttnn.deallocate(xr)
        x1 = self._col_sum(s)
        h = self._norm(x1)
        sbw = 4 if getattr(self.ck_ff13, "fp32_dest_acc_en", False) else 8
        if self.fused_swiglu:
            cfg = ttnn.MinimalMatmulConfig(
                M_block_size=1,
                K_block_size=8,
                N_block_size=8,
                subblock_h=1,
                subblock_w=sbw,
                compute_with_storage_grid_size=self.grid,
            )
            f = ttnn.experimental.minimal_matmul(
                input_tensor=h,
                weight_tensor=L["w13"],
                config=cfg,
                compute_kernel_config=self.ck_ff13,
                memory_config=DRAM,
                dtype=ttnn.bfloat8_b,
                fuse_swiglu=True,
            )
        else:
            f13 = ttnn.matmul(h, L["w13"], compute_kernel_config=self.ck_ff13, dtype=ttnn.bfloat8_b, memory_config=DRAM)
            f1, f3 = ttnn.split(f13, self.ff, dim=-1)
            ttnn.deallocate(f13)
            f = ttnn.mul(
                f1, f3, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], dtype=ttnn.bfloat8_b, memory_config=DRAM
            )
            ttnn.deallocate(f1), ttnn.deallocate(f3)
        ttnn.deallocate(h)
        p2 = ttnn.matmul(f, L["w2"], compute_kernel_config=self.ck_ff2_row, dtype=ttnn.bfloat16, memory_config=DRAM)
        ttnn.deallocate(f)
        ff = self._col_sum(p2)
        x2 = ttnn.add(x1, ff, dtype=ttnn.bfloat16, memory_config=DRAM)
        ttnn.deallocate(x1), ttnn.deallocate(ff)
        return x2

    def rows_to_logits_input(self, rows):
        """[1,1,32,8192] last-token rows (column c's token on its chips) -> the LM-head input [1,1,128,2048] per chip,
        column c's token at row 32 c (the tail of logits_input_device)."""
        allc = ttnn.all_gather(rows, 2, cluster_axis=1, memory_config=DRAM)  # [1,1,128,8192]
        out = ttnn.mesh_partition(allc, dim=3, cluster_axis=1, memory_config=DRAM)
        ttnn.deallocate(allc)
        return out

    def forward(self, tokens_cols, M, page_tables_cols, kv_cache, last_idx_cols=None):
        """Run all layers. tokens_cols [4, M] (padded), page_tables_cols [4, nb] (-1 rows: no user).
        Returns (x, owned): x = SP residual [1,1,M/8,8192] per chip (prompt c on column c); owned = the caller must
        deallocate x (False when x is a traced output buffer, reused by the next replay).
        Lengths in LLAMA_BH_PREFILL_DP4_TRACE (opt-in, default none) run as a captured trace after their first call."""
        assert M % (TP * 128) == 0, f"DP4xTP8 prefill length must be a multiple of 1024, got {M}"
        # Opt-in only: the DP4 forward is device-bound (traced 0.619 s vs untraced 0.611 s for 4 x 4K), and a captured
        # trace hands its intermediate buffers back to later allocations, which a replay then overwrites.
        trace_lens = {int(m) for m in os.environ.get("LLAMA_BH_PREFILL_DP4_TRACE", "").split(",") if m.strip()}
        prof_on = os.environ.get("LLAMA_BH_PREFILL_DP4_PROFILE", "0") == "1"
        self.rows_out = False
        if M not in trace_lens or kv_cache is None or prof_on:
            last_sel = None
            if self.trim_last and last_idx_cols and self.layout != "hidden" and M >= self.trim_min:
                last_sel = self._last_sel(M, last_idx_cols)
                self.rows_out = True
            x = self._forward_dev(
                ttnn.to_device(self._host_tokens(tokens_cols, M), self.mesh),
                M,
                self._page_table(page_tables_cols) if kv_cache is not None else None,
                kv_cache,
                free_inputs=True,
                last_sel=last_sel,
            )
            if last_sel is not None:
                for t in last_sel:
                    ttnn.deallocate(t)
            self._seen.add(M)
            return x, True
        host_tok = self._host_tokens(tokens_cols, M)
        host_pt = self._page_table(page_tables_cols, device=False)
        tr = self._traces.get(M)
        if tr is None:
            if M not in self._seen:
                # compile (and settle the SDPA chunk choice) outside the trace
                x = self._forward_dev(
                    ttnn.to_device(host_tok, self.mesh),
                    M,
                    ttnn.to_device(host_pt, self.mesh),
                    kv_cache,
                    free_inputs=True,
                )
                self._seen.add(M)
                return x, True
            tok_buf = ttnn.to_device(host_tok, self.mesh)
            pt_buf = ttnn.to_device(host_pt, self.mesh)
            ttnn.synchronize_device(self.mesh)
            tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
            out = self._forward_dev(tok_buf, M, pt_buf, kv_cache, free_inputs=False)
            ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
            self._traces[M] = tr = {"tid": tid, "tok": tok_buf, "pt": pt_buf, "out": out}
            logger.info(f"DP4xTP8 captured trace M={M}")
        else:
            ttnn.copy_host_to_device_tensor(host_tok, tr["tok"])
            ttnn.copy_host_to_device_tensor(host_pt, tr["pt"])
        ttnn.execute_trace(self.mesh, tr["tid"], cq_id=0, blocking=False)
        return tr["out"], False

    def _forward_dev(self, tt_tok, M, pt, kv_cache, free_inputs, last_sel=None):
        cos, sin = self._rope_mats(M)
        x = self._embed(tt_tok, M)
        if free_inputs:
            ttnn.deallocate(tt_tok)
        hidden = self.layout == "hidden"
        if hidden:
            xs = x
            x = self._seq_to_hidden(xs)
            ttnn.deallocate(xs)
        layer_fn = self.layer_hidden if hidden else self.layer
        prof_on = os.environ.get("LLAMA_BH_PREFILL_DP4_PROFILE", "0") == "1"
        for i in range(self.n_layers):
            self._prof = [] if (prof_on and i == min(1, self.n_layers - 1)) else None
            kw = {"last_sel": last_sel} if (last_sel is not None and i == self.n_layers - 1) else {}
            x = layer_fn(i, x, M, cos, sin, pt, kv_cache[i] if kv_cache is not None else None, **kw)
            if self._prof:
                p = self._prof
                logger.info(
                    f"DP4xTP8 PROFILE M={M} layer {i} total {1e3 * (p[-1][1] - p[0][1]):.2f} ms: "
                    + " ".join(f"{p[j][0]}={1e3 * (p[j][1] - p[j - 1][1]):.2f}" for j in range(1, len(p)))
                )
        self._prof = None
        if pt is not None and free_inputs:
            ttnn.deallocate(pt)
        return x

    def last_hidden(self, x, M, last_idx_cols):
        """Host copies of each column's last-token hidden row: dict col -> torch [8192] (float32)."""
        R = M // TP
        out = {}
        if self.rows_out:
            devs = ttnn.get_device_tensors(x)
            return {c: ttnn.to_torch(devs[c]).float().reshape(32, -1)[0] for c in last_idx_cols}
        if self.layout == "hidden":
            # row li of column c: the 8 chips of the column each hold 1024 of its columns
            for c, li in last_idx_cols.items():
                t0 = (int(li) // 32) * 32
                win = ttnn.slice(x, [0, 0, t0, 0], [1, 1, t0 + 32, int(x.shape[-1])])
                devs = ttnn.get_device_tensors(win)
                parts = [ttnn.to_torch(devs[r * COLS + c]).float().reshape(32, -1)[int(li) - t0] for r in range(TP)]
                ttnn.deallocate(win)
                out[c] = torch.cat(parts)
            return out
        for c, li in last_idx_cols.items():
            # slice the 32-row tile holding the token on device first: the host then reads 512 KB, not the whole
            # [R, 8192] residual of that chip
            r, j = divmod(int(li), R)
            j0 = (j // 32) * 32
            # bounds as device tensors: a static slice would compile one program per distinct token position
            if not hasattr(self, "_tail_start"):
                rep = ttnn.ReplicateTensorToMesh(self.mesh)
                self._tail_start = ttnn.from_torch(torch.zeros(4, dtype=torch.int32), device=self.mesh, mesh_mapper=rep)
                self._tail_end = ttnn.from_torch(torch.zeros(4, dtype=torch.int32), device=self.mesh, mesh_mapper=rep)
            for dev_t, vals in ((self._tail_start, [0, 0, j0, 0]), (self._tail_end, [1, 1, j0 + 32, self.dim])):
                ttnn.copy_host_to_device_tensor(
                    ttnn.from_torch(
                        torch.tensor(vals, dtype=torch.int32), mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh)
                    ),
                    dev_t,
                )
            win = ttnn.slice(
                input_tensor=x, starts=self._tail_start, ends=self._tail_end, slice_dim=2, num_devices=R // 32
            )
            t = ttnn.to_torch(ttnn.get_device_tensors(win)[r * COLS + c]).float().reshape(32, self.dim)
            ttnn.deallocate(win)
            out[c] = t[j - j0]
        return out

    def logits_input_device(self, x, M, last_idx_cols):
        """The LM-head input of a step built on device: [1,1,128,8192] in the TP32 prefill-output layout (sharded on
        dim 3 over the mesh columns), the last-token hidden row of column c's prompt at row 32 c, with no host read
        (last_hidden + logits_inputs make the host wait for the forward, idling the device for the step's tail).
        x: the SP residual [1,1,M/8,8192] per chip; last_idx_cols: {column: last token index}."""
        R = M // TP
        sel = torch.zeros(TP, COLS, 32, R)  # one-hot row vector on the chip that holds the token, zeros elsewhere
        for c, li in last_idx_cols.items():
            r, j = divmod(int(li), R)
            sel[r, c, 0, j] = 1.0
        t_sel = ttnn.from_torch(
            sel,
            device=self.mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=DRAM,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh, dims=(0, 1), mesh_shape=(8, 4)),
        )
        row = ttnn.matmul(t_sel, x, compute_kernel_config=self.ck_norm, dtype=ttnn.bfloat16, memory_config=DRAM)
        ttnn.deallocate(t_sel)  # row: [1,1,32,8192], row 0 = the token on its chip, zeros elsewhere
        col = ttnn.all_gather(row, 2, cluster_axis=0, memory_config=DRAM)  # [1,1,256,8192]
        ttnn.deallocate(row)
        col = ttnn.reshape(col, ttnn.Shape((1, TP, 32, self.dim)))
        one = ttnn.sum(col, dim=1, keepdim=True)  # [1,1,32,8192]: column c's token at row 0 on all its chips
        ttnn.deallocate(col)
        allc = ttnn.all_gather(one, 2, cluster_axis=1, memory_config=DRAM)  # [1,1,128,8192]: column c at row 32 c
        ttnn.deallocate(one)
        out = ttnn.mesh_partition(allc, dim=3, cluster_axis=1, memory_config=DRAM)  # [1,1,128,2048] per chip
        ttnn.deallocate(allc)
        return out

    def logits_inputs(self, hiddens):
        """List of [8192] host rows -> ONE [1,1,128,8192] device tensor (row 32 k = hiddens[k]) in the TP32
        prefill-output layout, so a step uploads once; pass last_token_idx = 32 k to the LM head for row k."""
        assert len(hiddens) <= 4
        t = torch.zeros(1, 1, 128, self.dim)
        for k, h in enumerate(hiddens):
            t[0, 0, 32 * k] = h
        return ttnn.from_torch(
            t,
            device=self.mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=DRAM,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh, dims=(None, 3), mesh_shape=(8, 4)),
        )

    def logits_input(self, hidden):
        """[8192] host row -> [1,1,128,8192] device tensor in the TP32 prefill-output layout (row 0 = the token).
        128 rows: the shortest prefill length the model's norm / LM-head collectives have buffers for."""
        t = torch.zeros(1, 1, 128, self.dim)
        t[0, 0, 0] = hidden
        return ttnn.from_torch(
            t,
            device=self.mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=DRAM,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh, dims=(None, 3), mesh_shape=(8, 4)),
        )
