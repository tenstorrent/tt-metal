# AutoDebug: Phi-3.5 optimized-decoder 8-core correctness collapse

## Scope and starting evidence

- Target: `microsoft/Phi-3.5-mini-instruct`, single-device optimized decoder.
- Reported failing policy:
  `decode_core_grid=(8,1)`, QKV/O/gate-up/down `in0_block_w=4`,
  BFP8 attention and MLP weights, HiFi2 math, and BF16 KV cache.
- Reported result: traced decode is approximately `0.765 ms` at batch 1
  and `0.947 ms` at batch 32, but final output PCC against
  `_reference_decode_zero_prefix` is only approximately `0.011` and `0.006`.
- Passing contrast: the default 32-core policy, `(8,4)` with
  `in0_block_w=3/3/3/8`, passes synthetic decode at approximately
  `0.999983` PCC.
- Resource contrast: an earlier 8-core QKV `in0_block_w=12` attempt failed
  L1 allocation.
- Inspected source:
  `tt/optimized_decoder.py`, `tt/functional_decoder.py`,
  `tests/test_functional_decoder.py`, common 1D attention/MLP/RMSNorm
  helpers, the DRAM-sharded matmul validation/program factory and kernels,
  and `nlp_create_qkv_heads_decode`.
- This investigation was source and artifact inspection only. It did not open
  a TT device or run hardware tests, and it did not edit implementation code.

The repo-local AutoDebug runner was invoked first as required, but its fresh
Codex process could not execute any source-read command because this host
disallows the user namespace required by its workspace sandbox:

```text
bwrap: No permissions to create a new namespace
```

The diagnosis below was therefore completed in this already-fresh forked
agent, with two independent read-only source passes. The report is the only
file written.

## Direct observations versus interpretations

Direct observations:

- Precision is not the failing-versus-passing delta. Both reported policies
  use BFP8 weights, HiFi2 math, BF16 activations/outputs, and BF16 cache.
- Cache, RoPE, paged update, and decode SDPA configuration do not depend on
  `decode_core_grid`; the SDPA grid remains `(8,8)`.
- Changing `decode_core_grid` changes the residual shard, both sharded
  RMSNorm calls, all projection input/output shards, DRAM-matmul output storage
  geometry, and which QKV-head program factory is selected.
- The failing policy also changes every material matmul's K blocking at once.
- Existing checked-in tests import and instantiate only `FunctionalDecoder`;
  they do not currently exercise the untracked optimized implementation or
  the 8-core policy.

Interpretations, not yet hardware-proven:

- The failure is unlikely to be ordinary BFP8 numerical loss; a PCC near zero
  points to wrong data, ordering, sharding, or a kernel/config contract.
- The first bad boundary is most likely either the 8x1 sharded RMSNorm or the
  immediately following QKV DRAM-sharded matmul. The final-output PCC alone
  cannot distinguish them.
- The packed MLP split is a separate later risk and must not be blamed until
  attention/post-attention boundaries pass.

## Instantiated geometry ledger

For decode, logical batch 1 is tile-padded to physical M=32, so
`per_core_M=1` is correct. The failing 8x1 policy lowers as follows:

| Boundary | Logical width | 8-core L1 width shard | Program geometry |
|---|---:|---:|---|
| residual / normalized | 3072 | `(32,384)` = 12 tiles | RMSNorm `block_w=12`, `subblock_w=4` |
| packed QKV output | 9216 | `(32,1152)` = 36 tiles | Kt=96, Nt=288, `in0=4`, `per_core_N=36` |
| O projection output | 3072 | `(32,384)` = 12 tiles | Kt=96, Nt=96, `in0=4`, `per_core_N=12` |
| packed gate/up output | 16384 | `(32,2048)` = 64 tiles | Kt=96, Nt=512, `in0=4`, `per_core_N=64` |
| activated MLP intermediate | 8192 | `(32,1024)` = 32 tiles | down input |
| down output | 3072 | `(32,384)` = 12 tiles | Kt=256, Nt=96, `in0=4`, `per_core_N=12` |

Source: `optimized_decoder.py:127-165`, `:313-337`.

The decisive K-block contrast is:

| Role | Passing 32-core input shard | Passing blocks/shard | Failing 8-core input shard | Failing blocks/shard |
|---|---:|---:|---:|---:|
| QKV | 3 K tiles, `in0=3` | 1 | 12 K tiles, `in0=4` | 3 |
| O | 3 K tiles, `in0=3` | 1 | 12 K tiles, `in0=4` | 3 |
| gate/up | 3 K tiles, `in0=3` | 1 | 12 K tiles, `in0=4` | 3 |
| down | 8 K tiles, `in0=8` | 1 | 32 K tiles, `in0=4` | 8 |

The DRAM-sharded factory computes on one optimal worker per DRAM bank, rather
than on the requested activation/output shard count
(`matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp:113-145`).
The requested grid instead controls activation storage/multicast senders and
output storage.

The factory derives:

```text
num_blocks_per_shard = (Kt / in0_block_w) / input_storage_cores
```

at `matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp:288-322`.
The input reader maps each global K block to a sender with integer division and
advances through consecutive local blocks at
`reader_bmm_tile_layout_in0_sender_dram_sharded.cpp:33-50` and `:73-234`.

The model-side check only proves that Kt and each activation shard width are
divisible by `in0_block_w` (`optimized_decoder.py:322-337`). The common matmul
validation similarly does not assert the derived block-to-input-core mapping
or the output shard shape/grid against `per_core_N`
(`matmul_device_operation.cpp:1341-1392`).

Existing nightly coverage is close but not exact:
`test_matmul_dram_sharded.py:278-293` covers 8x1 BFP8/HiFi2 geometries such as
K=4096 or 8192 with four blocks per shard. It does not cover this model's
K=3072, `in0=4`, three-block-per-shard hidden projections, nor their exact
large N widths and QKV-head consumer.

## Ranked hypotheses

### H1: the first 8x1 sharded RMSNorm is already wrong

This is the earliest geometry-sensitive arithmetic operation. Decode first
reshards the BF16 input to `(32,384)` over 8x1, then runs RMSNorm with
`block_w=12`, `subblock_w=4` (`optimized_decoder.py:662-665`,
`:382-390`). The passing path uses 8x4, `block_w=3`, `subblock_w=3`.

The lowered RMSNorm topology also changes:

- 8x1 is a one-stage 1D row-wise reduction.
- 8x4 enables the two-stage reduction path.

That follows from
`sharded_layernorm_factory_helpers.cpp:81-104` and `:112-173`.
The common RMSNorm helper constructs the same nominal program fields, so there
is no obvious Python configuration error. This remains a kernel/path
hypothesis, not a proven bug.

Prediction: the lossless DRAM-to-8x1-L1 round trip passes, but the first
`_norm_decode` output has low PCC or wrong magnitude. If both round trip and
norm pass, H1 is refuted and QKV is the first remaining boundary.

Smallest intervention if verified: keep or select a known-good norm shard grid
independently of the projection working grid, with one measured reshard at the
boundary, then retune the coherent layer path. Do not change global precision
as a “fix.”

### H2: QKV is corrupted by the 8-core DRAM-matmul storage/block contract

This is the strongest specialized-kernel differential after RMSNorm. QKV is
the first projection and changes from one to three K blocks per activation
shard. The same transition occurs for O and gate/up, while down changes from
one to eight.

There is a second QKV-specific differential inside the factory:

- On an 8-bank Blackhole, QKV compute width is `Nt/8 = 36` tiles regardless
  of output storage grid.
- The subblock heuristic internally pads the compute width from 36 to 40 and
  records four valid lanes in the last 8-wide subblock
  (`matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp:144-179`).
- With 8 output shards, each worker writes one 36-tile shard.
- With 32 output shards, each worker splits its 36 valid tiles across four
  9-tile storage shards.

Those are different writer branches at
`matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp:710-887`;
the kernel performs the resulting writeback at
`reader_bmm_tile_layout_in1_sender_dram_sharded.cpp:182-219`.
The passing 32-core result proves the common QKV compute and 36-to-40 padded
subblock can be correct. It does not prove the 8-shard equality/writeback
branch or the three-block activation sender path.

Prediction: RMSNorm passes, but packed QKV PCC collapses before head creation.
A K-boundary-sensitive failure implicates activation sender mapping; an
N-stripe/core-sensitive failure implicates output writeback.

Smallest intervention if verified: choose the fastest correct 8-core
`in0_block_w`/storage combination, or decouple activation and output storage
grids. A framework fix needs a minimal standalone matmul repro before changing
shared TTNN code.

### H3: 8-core QKV selects a different head-creation factory

`nlp_create_qkv_heads_decode` dispatches sharded inputs on a single rectangular
range to its regular sharded factory, while multi-range/non-origin inputs use
the subcore-grid factory
(`nlp_create_qkv_heads_decode.cpp:23-40` and
`nlp_create_qkv_heads_decode_device_operation.cpp:12-24`).

An 8-core QKV output is naturally the single range `(0,0)-(7,0)`. The
32-storage-core output generated on Blackhole's wider device grid can be a
multi-range CoreRangeSet, selecting the other factory. The regular kernel's
static arithmetic is internally consistent for this case:
`288 QKV tiles / 8 = 36 tiles/core`
(`reader_tm_tile_layout_nlp_create_qkv_heads_decode.cpp:72-133`).
Therefore this is lower confidence than H2.

Prediction: packed QKV itself passes, but raw Q/K/V heads are the first bad
outputs. Converting the packed QKV to an interleaved or deliberately
subcore-grid input before head creation changes the failure.

### H4: the packed gate/up split is the first late failure

`gate_up_split_interleaved` defaults to `False`. The 8-core packed result
therefore reaches `ttnn.split` as a `(32,2048)` width-sharded tensor
(`optimized_decoder.py:429-444`). The nearby implementation comment explicitly
documents that some few-core shard geometries cannot represent both 8192-wide
halves and provides a one-boundary DRAM-interleaved workaround.

Prediction: attention through the attention residual passes, gate/up linear
passes, but one split half or the activated product is wrong.
Setting only `gate_up_split_interleaved=True` restores final PCC.

This is a strong late-path candidate but cannot explain the first bad boundary
if normalized or QKV output is already wrong.

### H5: stale program/trace state makes a correct cold policy look wrong

The policy changes shard specs and program geometry while keeping the same
logical tensor shapes. TTNN is expected to include memory/program config in
the program hash, so source inspection did not prove a cache-key bug.

Prediction: the failure occurs only when 32-core and 8-core candidates run in
one process or reuse an earlier trace; it disappears in a cold process with a
fresh decoder, tensors, caches, and trace.

Run all localization candidates in separate processes first. Do not promote
this hypothesis unless cold-versus-warm order is a repeatable controlled
contrast.

## Required first-bad-boundary experiment

Run this as an ordinary correctness job, separately from watcher and profiler.
It checks the lossless reshard, first RMSNorm, packed QKV, post-attention
RMSNorm/gate-up, activated MLP value, and isolated full MLP under the exact
failing policy. It does not trace and does not invoke cache or SDPA.

```bash
python - <<'PY'
import torch
import torch.nn.functional as F
import ttnn

from models.autoports.microsoft_phi_3_5_mini_instruct.tests import test_functional_decoder as tf
from models.autoports.microsoft_phi_3_5_mini_instruct.tt.optimized_decoder import (
    OptimizedDecoder,
    OptimizationPolicy,
)
from models.common.utility_functions import comp_pcc

policy = OptimizationPolicy(
    decode_core_grid=(8, 1),
    qkv_in0_block_w=4,
    o_proj_in0_block_w=4,
    gate_up_in0_block_w=4,
    down_in0_block_w=4,
)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
try:
    config = tf._config()
    state = tf._synthetic_state(config)
    decoder = OptimizedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=tf.LAYER_IDX,
        mesh_device=mesh,
        batch=1,
        max_context=64,
        policy=policy,
    )
    hidden = torch.randn(
        1, 1, config.hidden_size, generator=torch.Generator().manual_seed(101)
    ).to(torch.bfloat16)
    tt_hidden = tf._to_tt_decode(hidden, mesh)

    def host(value):
        return ttnn.to_torch(ttnn.get_device_tensors(value)[0]).squeeze(0).transpose(0, 1)

    def report(label, reference, actual):
        passed, message = comp_pcc(reference.float(), actual.float(), 0.995)
        diff = (reference.float() - actual.float()).abs()
        print(
            f"BOUNDARY label={label} pass={passed} pcc={message} "
            f"max_abs={float(diff.max())} mean_abs={float(diff.mean())} "
            f"finite={bool(torch.isfinite(actual).all())}"
        )

    residual = ttnn.to_memory_config(tt_hidden, decoder.residual_memory_config)
    report("reshard_roundtrip", hidden, host(residual))

    norm = decoder._norm_decode(residual, decoder.weights["input_norm"])
    norm_ref = tf._rms_norm(
        hidden, state[tf._key("input_layernorm.weight")], config.rms_norm_eps
    ).to(torch.bfloat16)
    report("input_rmsnorm", norm_ref, host(norm))

    qkv = decoder._linear_decode(
        norm, "qkv", decoder.qkv_memory_config, decoder.attention_compute_kernel_config
    )
    qkv_ref = F.linear(norm_ref, state[tf._key("self_attn.qkv_proj.weight")])
    report("packed_qkv", qkv_ref, host(qkv))
    print("QKV_MEMCFG", qkv.memory_config(), "SHAPE", tuple(qkv.shape))

    post_norm = decoder._norm_decode(residual, decoder.weights["post_norm"])
    post_norm_ref = tf._rms_norm(
        hidden, state[tf._key("post_attention_layernorm.weight")], config.rms_norm_eps
    ).to(torch.bfloat16)
    report("post_rmsnorm", post_norm_ref, host(post_norm))

    gate_up = decoder._linear_decode(
        post_norm, "gate_up", decoder.gate_up_memory_config, decoder.mlp_compute_kernel_config
    )
    gate_up_ref = F.linear(post_norm_ref, state[tf._key("mlp.gate_up_proj.weight")])
    report("packed_gate_up", gate_up_ref, host(gate_up))
    gate, up = ttnn.split(gate_up, decoder.intermediate_size, dim=-1)
    activated = ttnn.multiply(
        ttnn.silu(gate), up, memory_config=decoder.intermediate_memory_config
    )
    gate_ref, up_ref = gate_up_ref.chunk(2, dim=-1)
    activated_ref = F.silu(gate_ref) * up_ref
    report("gate_split_activated", activated_ref, host(activated))

    mlp = decoder._mlp_decode(residual)
    mlp_ref = hidden + F.linear(
        activated_ref, state[tf._key("mlp.down_proj.weight")]
    )
    report("isolated_full_mlp", mlp_ref, host(mlp))
finally:
    ttnn.close_mesh_device(mesh)
PY
```

Interpret the first failing row literally:

1. `reshard_roundtrip`: data-movement/shard ordering.
2. `input_rmsnorm`: 8x1 RMSNorm.
3. `packed_qkv`: DRAM-sharded QKV matmul.
4. If all three pass, inspect raw Q/K/V heads next before cache/SDPA.
5. `post_rmsnorm` / `packed_gate_up` / `gate_split_activated` /
   `isolated_full_mlp`: late MLP localization.

Do not keep host reads in the measured implementation. This is a diagnostic
probe only.

## Focused policy matrix using the existing trace/reference test

The following command reuses the existing functional trace/reference harness
without editing it. Run one `PHI_SPEC` and one batch per cold process.

```bash
PHI_SPEC=8_default_blocks PHI_BATCH=1 python - <<'PY'
import os
import ttnn

from models.autoports.microsoft_phi_3_5_mini_instruct.tests import test_functional_decoder as tf
from models.autoports.microsoft_phi_3_5_mini_instruct.tt.optimized_decoder import (
    OptimizedDecoder,
    OptimizationPolicy,
)

specs = {
    "32_default": OptimizationPolicy(),
    "8_default_blocks": OptimizationPolicy(
        decode_core_grid=(8, 1),
        qkv_in0_block_w=3,
        o_proj_in0_block_w=3,
        gate_up_in0_block_w=3,
        down_in0_block_w=8,
    ),
    "8_block4": OptimizationPolicy(
        decode_core_grid=(8, 1),
        qkv_in0_block_w=4,
        o_proj_in0_block_w=4,
        gate_up_in0_block_w=4,
        down_in0_block_w=4,
    ),
    "8_block4_split_dram": OptimizationPolicy(
        decode_core_grid=(8, 1),
        qkv_in0_block_w=4,
        o_proj_in0_block_w=4,
        gate_up_in0_block_w=4,
        down_in0_block_w=4,
        gate_up_split_interleaved=True,
    ),
}
policy = specs[os.environ["PHI_SPEC"]]

class ProbeDecoder(OptimizedDecoder):
    @classmethod
    def from_state_dict(cls, *args, **kwargs):
        kwargs["policy"] = policy
        return super().from_state_dict(*args, **kwargs)

tf.FunctionalDecoder = ProbeDecoder
mesh = ttnn.open_mesh_device(
    ttnn.MeshShape(1, 1), trace_region_size=128 * 1024 * 1024
)
try:
    tf.test_decode_trace_replay_is_deterministic(mesh, int(os.environ["PHI_BATCH"]))
finally:
    ttnn.close_mesh_device(mesh)
PY
```

Run in this order:

```bash
PHI_SPEC=8_default_blocks PHI_BATCH=1 <the-command-above>
PHI_SPEC=8_block4_split_dram PHI_BATCH=1 <the-command-above>
```

The reported `32_default` and `8_block4` results need not be repeated unless
the earlier runs shared one process or stale trace state.

Interpretation:

- If `8_default_blocks` passes, the 8x1 grid and RMSNorm are viable; vary one
  material `in0_block_w` at a time to locate the failing matmul.
- If it fails and the boundary probe first fails RMSNorm, retune the norm grid.
- If it fails first at QKV, use the low-level 2x2 grid experiment below.
- If only `8_block4_split_dram` passes, the packed split is verified.

The unchanged `qkv_in0_block_w=12` experiment should not be rerun as the first
control because it already has a reproducible L1 OOM. Try the intermediate
legal QKV values 3, 4, and 6 before revisiting 12 with an explicitly reduced
L1 configuration.

## Low-level matmul discriminator

If `packed_qkv` is the first bad boundary, make a standalone real-shape
matmul test and cross input and output storage counts while holding BF16 x
BFP8, HiFi2, M=32, and weights fixed:

| Input storage | Output storage | QKV block | Purpose |
|---:|---:|---:|---|
| 32 | 32 | 3 | known-good control |
| 32 | 8 | 3 | isolates 8-shard/equality writeback |
| 8 | 32 | 3 | isolates multi-block input sender/multicast |
| 8 | 8 | 3, 4, 6 | integrated few-core candidates |

Use `per_core_N=9` for 32 QKV output shards and `36` for 8. For O/down use
`3` versus `12`; for gate/up use `16` versus `64`.

Then sweep:

- hidden projections `in0_block_w={3,4,6,12}`, producing
  `{4,3,2,1}` blocks per 8-core input shard;
- down `in0_block_w={8,4,16,32}`, producing
  `{4,8,2,1}` blocks per shard.

If only hidden `in0=4` fails, the odd three-block sender path is the decisive
contrast. If 32-input/8-output fails, investigate writeback. If
8-input/32-output fails, investigate input sender/multicast. Keep each run in
a cold process until program-cache reuse has been ruled out.

For the failing standalone shape, a one-hot K-block/marked N-stripe follow-up
is more informative than another full-layer PCC:

- probe hidden K blocks around 2/3 and 5/6 for `in0=4`;
- probe down K blocks around 7/8 and 15/16;
- mark weight columns by DRAM-bank/output-shard stripe.

A K-boundary permutation identifies input sender mapping. An N-stripe/core
permutation identifies output writeback.

Also log the effective runtime contract:

```text
logical/padded input and weight shapes
input/output shard grid, shard shape, orientation, and buffer
weight DRAM grid and shard shape
program config and compute config
num_blocks and num_blocks_per_shard
per_core_N sender / padded compute / output storage
active worker/storage bounding box versus full device grid
```

The factory currently forms multicast coordinates/counts from the full device
compute grid while its CB/kernel/semaphore ranges come from the bounding box
of activation storage plus optimal DRAM readers
(`matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp:91-101`,
`:270-281`, `:301-322`, and `:399-631`). If those sets differ on the failing
device, preserve the descriptor dump as a minimal framework repro.

## Final diagnosis

Source inspection does not justify a code fix yet. The final PCC combines too
many simultaneous geometry changes to name the first corrupting operation.

The actionable diagnosis is:

1. The earliest boundary to clear is the DRAM-to-8x1 residual reshard and first
   sharded RMSNorm.
2. If those pass, packed QKV is the first and strongest suspect. The failing
   policy enters an exact three-K-block-per-input-shard path and an 8-shard
   output-writeback contract that the passing policy does not exercise.
3. If QKV and attention pass, test the documented few-core packed gate/up
   split workaround before changing precision or cache policy.

Do not reject 8-core DRAM-sharded decode based on the current final-output PCC.
The reported latency proves the geometry runs quickly, while the boundary
experiments above determine whether a correct variant exists or whether a
minimal TTNN matmul/head/split repro is required.
