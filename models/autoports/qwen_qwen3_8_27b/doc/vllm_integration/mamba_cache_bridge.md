# Hybrid prefix-cache bridge

Implemented against standalone plugin1badbd02bb6911dfdb59c206bdc00c6c1e8dc0bf
and installed upstream vLLM0.30.0+empty. This is host interface evidence;
device snapshot/restore correctness and memory fit remain parent-stage work.

The existing upstream MambaSpec/MambaManager align path owns prefix hashes,
shared snapshot lifetime, block IDs and eviction. The model declares one raw
layer-indexed spec per64 layers:16 FullAttentionSpec and48 GDN MambaSpec.
The planner forms4 groups of16 layers. Attention and Mamba boundaries are4096
tokens; full attention expands each scheduler block to128 internal32-token pages.
No new plugin capability, environment variable, or additional-config key was added.

Plugin changes:
- MambaSpec groups are accepted and passed as the exact KVCacheConfig to
  `allocate_hybrid_kv_cache(config)`; its opaque return retains ordinary ownership.
- Mixed attention/recurrent `page_tables_per_layer` uses all64 raw model indices,
  rather than16 compact attention ordinals. Attention-only consumers retain their
  existing indexing and allocation API.
- Pool sizing reserves2 recurrent IDs per request per Mamba group, plus declared
  checkpoint/speculative blocks and the shared null block. With native262144
  attention token budget and4096 block size, totals are72 IDs atB1 and289 atB32.
- Runtime interface documentation accompanies the changes.

Model construction helper: `tt/vllm_cache_specs.py`. Attention's BF16/global-head
bytes are virtual planner units; Mamba pages are virtually padded to that unit
to preserve4096 block size. Device buffers must use selected BFP8 attention plus
BF16 convolution/FP32 recurrence; physical memory is accounted separately.

Per device, each linear layer snapshot is801792 bytes:
conv3x2560x2=15360 plus recurrence12x128x128x4=786432.
All48 layers total38486016 bytes per global block ID. With289 IDs, typed
recurrent pools use11122458624 bytes and BFP8 attention pools use10303307776
bytes (289x4096x8704):21425766400 bytes total, excluding live recurrence,
weights, workspaces, trace and scratch. Live B32 recurrence adds1231552512
bytes. These are shape calculations, not measured allocator peaks or proof of fit.
The32x32768 target cannot simply assume this pool is all attention capacity:
192 IDs can be occupied by the three recurrent groups' live states. Record
actual pool pressure/shortfall and allocator peaks without reducing native context.

Exact checkpoint contract: state after `end` tokens belongs to the per-layer
table entry `(end-1)//4096`. A cached aligned prefix starting at`start>0` restores
entry`(start-1)//4096`. Save only advertised aligned boundaries, preserve active
slot state between them, and never mutate a shared cached state while advancing
a request. A new/resumed request must reset or restore; a continuing prefill may
retain its current slot. Decoder positions advance on device, so stale async
host positions cannot decide whether a boundary snapshot is due. State-row DMA
and its indices/scratch must be allocated and warmed before any live trace.

Validation command (no hardware; import-only TTNN stubs):
`docker exec -w /workspace/bringup/dependencies/vllm-tt-plugin -e USER=mvasiljev -e PYTHONPATH=tests/host-stubs -e TORCHINDUCTOR_CACHE_DIR=/workspace/.cache/torchinductor qwen38-functional-build /opt/venv/bin/python -m pytest --confcutdir=. tests/test_mamba_cache_bridge.py tests/test_model_runner.py tests/test_num_available_blocks.py tests/test_worker_lifecycle.py tests/test_state_slots.py -q`

Result:67 passed,2 SWIG deprecation warnings. Cases cover exact opaque allocation,
missing hybrid allocator rejection, raw64 table mapping, real upstream grouping
and B1/B32 budget calculations, existing attention/hybrid sizing, runner behavior
and continuing-slot fixes. Log:`readiness_vllm/mamba_bridge_tests.log`.

Explicit pre-commit on changed plugin files passes (whitespace, conflict checks,
ruff check/format); log:`readiness_vllm/mamba_bridge_precommit.log`.
No commit or hardware execution was performed by this subtask.
