import sys

sys.path.insert(0, "/home/ttuser/sdawle/skills/blaze-vs-ttnn-bench/scripts")
import ab_harness

# MUST precede the ttnn import -- the harness enables the device profiler via env, and
# _measure() silently returns None (not an error) when no profiler samples come back.
ab_harness.set_profiler_env()

import torch, ttnn

# conftest normally registers this virtual namespace; a bare script must do it itself.
from blaze.models.blaze_tests_namespace import register_blaze_tests_namespace

register_blaze_tests_namespace()
from blaze.fused_program import FusedProgram
from blaze.ops.glm_qkv_a_projection import GLMQKVAProjection
from blaze_tests.micro_ops.common.test_dram_streaming_matmul import _make_weights_tensor, _pad_to_dram_banks
from models.demos.deepseek_v3.utils.config_helpers import (
    dram_sharded_weight_config,
    get_activation_sharding_core_counts_for_dram_matmul,
    get_dram_sharded_matmul_config,
)

d = ttnn.open_device(device_id=0)
K, NQ, NKV, DS = 2048, 768, 576, 32
banks = d.dram_grid_size().x
# A worker's shard must be a whole number of 32-wide tiles, so with W workers per bank N has to
# be a multiple of 32*banks*W. This is a real cost of widening at GLM's shapes: q_a 768 and
# kv_a 576 both pad to 1024 at W>=2, work ttnn does not do.
_W = int(__import__("os").environ.get("BLAZE_DSM_WORKERS_PER_BANK", "1"))
_PADM = 32 * banks * _W
nq, nkv = _pad_to_dram_banks(NQ, 32, _PADM), _pad_to_dram_banks(NKV, 32, _PADM)
torch.manual_seed(47)
row = torch.randn(1, 1, 1, K).bfloat16().float()
native = torch.zeros(1, 1, 32, K, dtype=torch.bfloat16).float()
native[..., :1, :] = row
wq = torch.randn(1, 1, K, nq).bfloat16().float()
wkv = torch.randn(1, 1, K, nkv).bfloat16().float()
# ---- blaze: native model tensor in, DRAM out
tt_nat = ttnn.from_torch(
    native, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=d, memory_config=ttnn.DRAM_MEMORY_CONFIG
)
b_wq = _make_weights_tensor(d, wq, k=K, n_padded=nq, tile_w=32, num_banks=banks, weight_dtype=ttnn.bfloat8_b)
b_wkv = _make_weights_tensor(d, wkv, k=K, n_padded=nkv, tile_w=32, num_banks=banks, weight_dtype=ttnn.bfloat8_b)
mk = lambda w: ttnn.from_torch(
    torch.zeros(1, 1, 1, w),
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=d,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
tq, tkv = mk(nq), mk(nkv)


# HOIST the FusedProgram build out of the timed callable. Rebuilding it per iteration measures
# host-side composition, not kernel time -- and ttnn pays no equivalent cost because it caches
# programs. Timing the rebuild is what produced the bogus 0.53x.
def _build():
    f = FusedProgram(
        kernel=None,
        device=d,
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        name="native_ab",
    )
    GLMQKVAProjection.emit(f, tt_nat, b_wq, b_wkv, q_a_out=tq, kv_a_out=tkv, prefix="qkv_a", fp32_dest_acc_en=True)
    return f


_PROG = _build()


def blaze_fn():
    _PROG.run()
    return tq, tkv


# ---- ttnn: the shipping fused q_kv_a (one 2048x1344 matmul), same native input
g = d.compute_with_storage_grid_size()
mx = g.x * g.y
n_cat = _pad_to_dram_banks(NQ + NKV, 32, 32 * banks)
w_cat = torch.cat([wq[..., :NQ], wkv[..., :NKV]], dim=-1)
if w_cat.shape[-1] < n_cat:
    w_cat = torch.nn.functional.pad(w_cat, (0, n_cat - w_cat.shape[-1]))
in_c = max(get_activation_sharding_core_counts_for_dram_matmul(K, mx))
out_c = max(get_activation_sharding_core_counts_for_dram_matmul(n_cat, mx))
t_w = ttnn.from_torch(w_cat, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT).to(
    d, dram_sharded_weight_config(K, n_cat, d.dram_grid_size())
)
t_pc = get_dram_sharded_matmul_config(m=DS, k=K, n=n_cat, input_num_shards=in_c, output_num_shards=out_c)
t_ckc = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True
)
act_mc = ttnn.create_sharded_memory_config_(
    shape=(DS, K // in_c),
    core_grid=ttnn.num_cores_to_corerangeset(in_c, ttnn.CoreCoord(g.x, g.y), row_wise=True),
    strategy=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
    tile_layout=True,
    use_height_and_width_as_shard_shape=True,
)


def ttnn_fn():
    a = ttnn.to_memory_config(tt_nat, act_mc)  # model resharding it pays today
    o = ttnn.linear(
        a, t_w, program_config=t_pc, memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG, compute_kernel_config=t_ckc
    )
    return ttnn.to_memory_config(o, ttnn.DRAM_MEMORY_CONFIG)


# run() mutates program state (_prepare_for_build / _compaction_applied), so reuse is not
# assumed -- verify the 2nd and 3rd run still produce the 1st run's numbers before timing.
try:
    blaze_fn()
    _r1 = (ttnn.to_torch(tq).clone(), ttnn.to_torch(tkv).clone())
    for _ in range(2):
        blaze_fn()
    _r3 = (ttnn.to_torch(tq), ttnn.to_torch(tkv))
    _ok = all(torch.allclose(a, b) for a, b in zip(_r1, _r3))
    print(f"RESULT program-reuse stable across 3 runs: {_ok}", flush=True)
    assert _ok, "repeated run() diverges -- hoisting is invalid, results below would be garbage"
except Exception as e:
    print("RESULT REUSE-FAIL:", str(e).split(chr(10))[0][:160], flush=True)
    ttnn.close_device(d)
    raise SystemExit


# Correctness gate against the real ttnn path, not a torch golden: if blaze and the shipping
# op agree on the same inputs, the fused op is a valid substitute for it.
def _pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-12))


try:
    ttnn_out = ttnn.to_torch(ttnn_fn())[..., :1, :]
    blaze_fn()
    q_pcc = _pcc(ttnn_out[..., :NQ], ttnn.to_torch(tq)[..., :1, :NQ])
    kv_pcc = _pcc(ttnn_out[..., NQ : NQ + NKV], ttnn.to_torch(tkv)[..., :1, :NKV])
    print(f"RESULT PCC vs ttnn shipping path: q_a={q_pcc:.6f} kv_a={kv_pcc:.6f}", flush=True)
    if min(q_pcc, kv_pcc) < 0.99:
        print("RESULT GATE FAILED -- timings below are meaningless", flush=True)
except Exception as e:
    print("RESULT PCC-FAIL:", str(e).split(chr(10))[0][:150], flush=True)

try:
    t_us, _ = ab_harness._measure(d, ttnn_fn, warmup=2, iters=5)
    b_us, _ = ab_harness._measure(d, blaze_fn, warmup=2, iters=5)
    print(f"RESULT ttnn  fused q_kv_a (native in, DRAM out) = {t_us} us", flush=True)
    print(f"RESULT blaze GLMQKVAProjection (native in, DRAM out) = {b_us} us", flush=True)
    if t_us and b_us:
        print(
            f"RESULT speedup {t_us/b_us:.2f}x | x47 layers {(t_us-b_us)*47/1000:+.2f} ms/token (upper bound)",
            flush=True,
        )
    else:
        print("RESULT INCOMPLETE: profiler returned no samples for one side", flush=True)
except Exception as e:
    print("RESULT FAIL:", str(e).split(chr(10))[0][:140], flush=True)
ttnn.close_device(d)
