# DeepSeek V3 B1 — Sharded NVMe Weight Cache

## Background

### Model structure

DeepSeek V3 has **61 decoder layers** (indices 0–60):

- **Layers 0–2 (dense):** Multi-Latent Attention (MLA) + standard dense
  MLP (gate/up/down projections, hidden size 18432).
- **Layers 3–60 (MoE):** Same MLA attention + Mixture-of-Experts FFN
  with 256 routed experts, 1 shared expert, and a gating network
  (per-expert hidden size 2048).

Plus an **embedding layer** and an **LM head** (RMSNorm + projection +
sampling).

### Pipeline topology

The production deployment runs as a **ring pipeline** of 64 stages
across **16 hosts**. Each stage is a separate OS process with its own
**4×2 mesh** (8 Tenstorrent devices). Each host runs **4 stages**
(4 processes sharing the host's CPU, memory, and NVMe). Data flows
between stages as activations (device-to-device) or tokens
(host-to-device / device-to-host at the boundaries).

```
64-process single-pod deployment (16 hosts × 4 stages/host):

  Host 0:   Stage 0  ── Embedding (H2D + lookup)
            Stage 1  ── Dense decoder layer 0
            Stage 2  ── Dense decoder layer 1
            Stage 3  ── Dense decoder layer 2
  Host 1:   Stage 4  ── MoE decoder layer 3
            Stage 5  ── MoE decoder layer 4
            Stage 6  ── MoE decoder layer 5
            Stage 7  ── MoE decoder layer 6
    ⋮            ⋮
  Host 15:  Stage 60 ── MoE decoder layer 59
            Stage 61 ── MoE decoder layer 60
            Stage 62 ── LM head (RMSNorm + matmul + argmax)
            Stage 63 ── Token passthrough (→ stage 0 for D2H)
```

Each decoder stage loads **only its own layer's weights** at startup.
The embedding stage loads the embedding table; the LM head stage loads
the projection matrix and final norm. Passthrough stages load nothing.
Within a host, all 4 stages load their weights in parallel, sharing
the host's NVMe bandwidth.

Smaller configurations exist for development (4-process single-galaxy,
16-process single-pod subset) where a subset of layers are loaded and
the rest are activation or token passthroughs.

### Weight preprocessing pipeline

The source weights are HuggingFace safetensors stored on shared NFS.
Before they can run on device, each layer's weights go through a
multi-step preprocessing pipeline:

```
HF safetensors (NFS)
  │
  ▼
1. Extract ─────────── LazyStateDict reads only the needed tensors
  │                     from the relevant safetensors shard.
  ▼
2. Torch transforms ── Transpose to matmul layout, reorder q_b_proj
  │                     columns from per-head interleaved
  │                     [h0_nope|h0_rope|h1_nope|…] to grouped
  │                     [ALL_NOPE|ALL_ROPE], split kv_b_proj into
  │                     kv_b1 + kv_b2 (V half transposed), reshape
  │                     norms to (1, W).
  ▼
3. Device transforms ─ Tensor-parallel slicing across the 4x2 mesh,
  │                     quantization (bf16 → bfp8 or bfp4),
  │                     tile-shape conversion, shard reordering
  │                     (e.g. DRAM bank-aware tile shuffle for experts).
  ▼
4. Fuse ───────────── Pack multiple related tensors into single
  │                     overlapped buffers for L1 locality. Four
  │                     fusion groups per MoE layer: q_ab_kv_a,
  │                     o_proj_gate_mm_norms, kv_b12, gate_up.
  ▼
5. Place on device ─── ttnn.from_torch → device DRAM or L1 with
                        the target MemoryConfig and sharding strategy.
```

**Why this is expensive:** Steps 1–4 take ~210–245 seconds for a
single MoE layer (dominated by reading ~21 GB of bf16 safetensors from
NFS and the compute-heavy transforms). The result is deterministic for
a given model revision, mesh shape, and preprocessing version.

### Per-layer weight inventory

**MoE layer** (58 layers, each — measured from `cache-2026-03-22/layer_003/`):

| Component | Files | Cached size | Format | Placement |
|-----------|-------|-------------|--------|-----------|
| q_ab_kv_a (q_a, q_b, kv_a) | 1 | 72 MB | bfp8, WIDTH_SHARDED | L1 |
| o_proj_gate_mm_norms (o_proj, gate_mm, 4 norms) | 1 | 130 MB | mixed dtype, WIDTH_SHARDED | L1 |
| kv_b12 (kv_b1, kv_b2) | 1 | 18 MB | bfp8, HEIGHT_SHARDED | L1 |
| gate_up (shared gate, shared up) | 1 | 16 MB | bfp4, HEIGHT_SHARDED | L1 |
| shared_down_proj | 1 | 7.9 MB | bfp4 | L1 |
| gate_bias | 1 | 8 KB | bf16 | L1 |
| 256 routed experts × 3 projections | 768 | **6.0 GB** | bfp4, WIDTH_SHARDED | DRAM |
| **Total per MoE layer** | **774** | **6.2 GB** | | |

Routed experts dominate: **97% of cache size** and **99% of file
count** per MoE layer. Each expert projection is ~7.9 MB (bfp4,
padded and DRAM-tile-shuffled).

**Dense layer** (3 layers — measured from `layer_000/`):

| Component | Files | Cached size |
|-----------|-------|-------------|
| Attention fusion groups (q_ab_kv_a, o_proj_gate_mm_norms, kv_b12) | 3 | 220 MB |
| MLP (gate_up, shared_down, routed gate/up/down) | 5 | 216 MB |
| **Total per dense layer** | **8** | **432 MB** |

Dense layers have no MoE router and no 256-expert fan-out; the MLP is
a single set of gate/up/down projections (intermediate size 18432 vs
2048 per expert). For device mapping, the dense MLP is decomposed into
the same shared + routed structure used by MoE layers: the first 2048
columns (gate/up) or rows (down) form a "shared expert" placed in L1
identically to the MoE shared expert, and the remaining 8 × 2048 are
split into 8 routed experts, one per device in the 4×2 mesh (placed
in DRAM). This is why `DeepSeekV3DenseLayerWeights` has both
`shared_gate_proj` and `routed_gate_proj` fields.

**HF source size per MoE layer** (~21.4 GB in bf16): attention
~357 MB, shared expert ~84 MB, 256 routed experts ~21.5 GB, gate
~3.5 MB, norms negligible. The 3.5× reduction to 6.2 GB comes from
quantization (bf16 → bfp4 for experts ~4×, bf16 → bfp8 for attention
~2×) plus tiling/padding overhead.

**Full model cache** (measured, `cache-2026-03-22/`): **362 GB**

| Component | Layers | Cached size |
|-----------|--------|-------------|
| MoE layers | 58 | 58 × 6.2 GB = 360 GB |
| Dense layers | 3 | 3 × 432 MB = 1.3 GB |
| Embedding | 1 | 1.8 GB |
| LM head + final norm | 1 | 940 MB |
| **Total** | | **362 GB** |

### The problem

Without caching, every startup pays the full preprocessing cost per
layer. In the 64-process pipeline, all stages load their weights **in
parallel** — 4 stages per host sharing the host's NVMe, 16 hosts
sharing the NFS server — so wall-clock time is dominated by the
slowest host's load time, not the sum of all layers.

Parallel loading compounds the NFS problem: 64 processes (4 per host
× 16 hosts) all reading ~21 GB each from the same NFS server
simultaneously creates severe contention. The single-process
benchmarks below (which already show NFS struggling at 56 s per
layer) do not capture this multi-reader penalty.

The current caching system (see "Current State and Migration Plan"
below) solves this partially but has no content addressing, no
integrity verification, duplicated fusing logic, and a separate
offline generation step. In a multi-developer environment — where
several engineers are iterating on preprocessing transforms, mesh
configurations, or model revisions simultaneously — the lack of
content addressing makes cache management a manual, error-prone
process: developers must coordinate version bumps, manually
regenerate and validate caches, and risk silently using stale
artifacts from each other's changes.

---

## Content-Addressed Tensor Cache

The rest of this document describes a replacement for the current
caching system. The goal is a single interface —
`cache.get_or_create(fingerprint, ...)` — that transparently handles
the full lifecycle: on a cache hit it loads the preprocessed tensor
from the host's local NVMe in ~12 seconds; on a miss it reads from
NFS, runs the preprocessing pipeline, stores the result to NVMe, and
returns the tensor. No manual versioning, no offline validation step.
An offline pre-warming script is retained for production deployment
but is no longer required — any cache miss is handled transparently.

While the immediate motivation is DeepSeek V3, the core cache
infrastructure — fingerprinting, content-addressed storage, and
two-tier NVMe/NFS layout — is **model-agnostic**. Any model that
preprocesses HuggingFace weights into device-specific formats can use
the same `TensorCache` and `TensorTarget` types. The model-specific
parts are the fingerprint fields (source tensor names, transform
version) and the preprocessing callback, both of which are supplied
by the caller, not baked into the cache.

A key motivation beyond performance is **developer productivity**.  Today, the
cache is managed manually: the developer is responsible for identifying when the
cache is stale and needs to be regenerated, which is very error-prone. This
issue is compounded when many different developers are trying to work on
different features in parallel. Content addressing removes this friction. Each
combination of source weights, transforms, and target layout hashes to a unique
artifact — so multiple developers can iterate on preprocessing, mesh
configurations, or model revisions in parallel without coordinating
invalidation. The right artifact is always selected automatically, or rebuilt
transparently on first use. No need to coordinate "did you regenerate the
cache?", no stale-data debugging sessions, no manual versioning.

The design is built around three ideas:

1. **Content addressing.** Every cached artifact is keyed by a SHA-256
   fingerprint that captures the source tensors, model revision,
   preprocessing version, mesh shape, and target layout. Same
   fingerprint = same bytes. No ambient state, no version integers
   that can drift.

2. **Declarative layout specs.** Standalone tensors are described by
   `TensorTarget` (dtype, layout, memory config, tile shape, mesh
   mapping strategy). Planned: the four fusion group layouts will be
   expressed as `FusionGroupSpec` data, and a single generic function
   will pack any spec into a fused buffer, replacing four bespoke
   methods with one.

3. **Two-tier storage.** HuggingFace safetensors live on shared NFS
   (read only on cache miss). Preprocessed artifacts live exclusively
   on per-host NVMe — the `TensorCache` reads and writes only to
   local NVMe, never to shared storage. This keeps cold starts fast
   and NFS traffic near zero in steady state; the only NFS access is
   reading the source HF weights when an artifact is missing from the
   local cache.

---

## Core Concepts

### Fingerprint

The cache key. Captures everything that must match for a cached artifact to
be usable:

- Which source weights (HF tensor names)
- From which model revision
- Which preprocessing version
- Which mesh topology
- Which target format (spec or tensor target)

All fields are known before any computation runs.

### Artifact

The cached result. An immutable blob (the fused tensor bytes) plus metadata
(computed view properties, content hash, size). Stored in a
content-addressed object store keyed by `sha256(canonical(fingerprint))`.

### Spec

The target layout description. For standalone tensors, a `TensorTarget`
describes the full `from_torch` parameters (dtype, layout, memory config,
tile shape, and mesh mapping strategy via `MeshMapperConfig`). For fusion
groups, a `FusionGroupSpec` declaratively describes which sub-tensors go
on which cores, with which dtypes and tile shapes. The spec is part of
the fingerprint — any change in target layout produces a different
artifact ID.

### Preprocessing Function

A model-specific callback `f(raw_tensors) -> preprocessed_tensors` that
applies shuffles, packing, TP slicing, etc. It is not serialized or
hashed — it is versioned by `transform_version` in the fingerprint, and
only invoked on a cache miss.

---

## Architecture Diagram

### Standalone tensor path (implemented)

```
    Caller                          TensorCache
      │                                 │
      │  get_or_create(fingerprint,     │
      │    preprocess, raw_tensors)     │
      │────────────────────────────────►│
      │                                 ├── artifact_id = sha256(canonical(fingerprint))
      │                                 ├── lookup on local NVMe
      │                                 │
      │                          ┌──────┴──────┐
      │                          │   STATE?    │
      │                          └──┬────┬──┬──┘
      │                    PRESENT/ CORRUPT| \ABSENT
      │                          /    |    \
      │               load from    delete   raw_tensors()  ◄─ lazy, only on miss
      │                 disk      entry       │
      │                  │          │     preprocess()      ◄─ torch transforms
      │                  │          │         │
      │                  │          │     from_torch()      ◄─ TensorTarget params
      │                  │          │         │
      │                  │          └──► store artifact     ◄─ atomic write
      │                  │                    │
      │                  │               load from disk
      │                  │                    │
      │                  └────────┬───────────┘
      │                           │
      │ ◄─────────────────────────┘
      │   ttnn.Tensor (on device)
```

### Full path with fusion groups (implemented)

On a cache miss, when `fingerprint.target` is a `FusionGroupSpec`,
`TensorCache.get_or_create` calls `create_overlapped_tensor(spec,
preprocessed, device)` (see `tensor_cache/fuse.py`, shim to `weights/fusion_runtime.py`), which delegates to
the existing `BlitzDecodeWeights.get_tt_*` packers for the four production
groups. It persists the fused buffer plus per-view metadata in
`metadata.json` and returns `dict[str, OverlappedTensor]` on load.

---

## Type Model

> **Implementation status:** Standalone and fusion-group types live in
> `tensor_cache/types.py`. View metadata for fusion entries is serialized
> to JSON in `metadata.json` via `tensor_cache/overlapped_metadata.py`
> (not a separate `OverlappedViewMeta` dataclass in `types.py`).

### Types (standalone tensors)

```python
@dataclass(frozen=True)
class SourceTensorSelection:
    """Which HF state dict tensors feed into this artifact."""
    names: tuple[str, ...]


@dataclass(frozen=True)
class ReplicateMeshMapper:
    """Replicate the tensor on every device in the mesh."""
    strategy: Literal["replicate"] = "replicate"

@dataclass(frozen=True)
class ShardMeshMapper:
    """Shard the tensor along a single dimension across the mesh."""
    dim: int
    strategy: Literal["shard"] = "shard"

@dataclass(frozen=True)
class Shard2dMeshMapper:
    """Shard the tensor along two dimensions across a 2D mesh."""
    dims: tuple[int | None, int | None]
    strategy: Literal["shard_2d"] = "shard_2d"

MeshMapperConfig = ReplicateMeshMapper | ShardMeshMapper | Shard2dMeshMapper


@dataclass(frozen=True)
class TensorTarget:
    """Complete specification for a single (non-fused) cached tensor artifact.

    Contains every parameter needed for ttnn.from_torch on the miss path.
    All fields participate in the fingerprint hash, including the mesh
    mapper config — so a change in sharding strategy produces a different
    artifact ID.
    """
    kind: Literal["tensor"] = "tensor"
    name: str = ""
    dtype: ttnn.DataType = ttnn.bfloat16
    layout: ttnn.Layout = ttnn.TILE_LAYOUT
    memory_config: ttnn.MemoryConfig = field(
        default_factory=lambda: ttnn.DRAM_MEMORY_CONFIG
    )
    tile_shape: tuple[int, int] = (32, 32)
    mesh_mapper_config: MeshMapperConfig = field(
        default_factory=ReplicateMeshMapper
    )


@dataclass(frozen=True)
class Fingerprint:
    """Cache key. All fields are known before any computation runs."""
    schema_version: int
    source: SourceTensorSelection
    hf_model_id: str
    hf_revision: str
    transform_version: int
    mesh_shape: tuple[int, int]
    target: TensorTarget | FusionGroupSpec  # ArtifactTarget in types.py


@dataclass(frozen=True)
class CacheContext:
    """Bundles the common cache key fields shared across all tensors
    in a model.

    Prepare functions accept this to avoid repeating hf_model_id,
    hf_revision, etc. for every standalone tensor they cache.
    """
    schema_version: int
    hf_model_id: str
    hf_revision: str
    transform_version: int
    mesh_shape: tuple[int, int]

    def fingerprint(
        self,
        *,
        source: SourceTensorSelection,
        target: TensorTarget | FusionGroupSpec,
    ) -> Fingerprint:
        return Fingerprint(
            schema_version=self.schema_version,
            source=source,
            hf_model_id=self.hf_model_id,
            hf_revision=self.hf_revision,
            transform_version=self.transform_version,
            mesh_shape=self.mesh_shape,
            target=target,
        )
```

### Artifact Metadata (stored, NOT hashed)

There are no dedicated `Manifest` / `Metadata` Python dataclasses in the
implementation. On each store, `TensorCache._store()` writes `manifest.json`
and `metadata.json` as inline JSON dicts (fingerprint canonical form,
`logical_name`, `artifact_id`, `content_hash`, `size_bytes`, `created_at`).

### Cache Types

`CacheEntry` is a discriminated union — each variant structurally
guarantees the fields it carries (no optional `paths` to check at
runtime):

```python
@dataclass(frozen=True)
class ContentAddressedStoragePaths:
    object_dir: Path
    data_path: Path


@dataclass(frozen=True)
class AbsentCacheEntry:
    artifact_id: str

@dataclass(frozen=True)
class PresentCacheEntry:
    artifact_id: str
    paths: ContentAddressedStoragePaths

@dataclass(frozen=True)
class CorruptCacheEntry:
    artifact_id: str
    paths: ContentAddressedStoragePaths

CacheEntry = AbsentCacheEntry | PresentCacheEntry | CorruptCacheEntry
```

### Fusion group types (implemented)

The following dataclasses describe fused-buffer layouts for fingerprinting
and cache targets. Constants such as `Q_AB_KV_A_SPEC` are defined in
`blitz_decode_weights.py` from the existing single-device overlap specs.

```python
@dataclass(frozen=True)
class SubTensorSpec:
    """One logical tensor within a fused buffer region."""
    name: str
    tensor_shape: tuple[int, int]
    dtype: ttnn.DataType
    tile_shape: tuple[int, int]


@dataclass(frozen=True)
class RegionSpec:
    """Sub-tensors sharing a core range, stacked per core."""
    core_range_set: ttnn.CoreRangeSet
    subtensors: tuple[SubTensorSpec, ...]


@dataclass(frozen=True)
class FusionGroupSpec:
    """Complete packing layout for an overlapped tensor group."""
    kind: Literal["fusion_group"] = "fusion_group"
    name: str = ""
    regions: tuple[RegionSpec, ...] = ()
    sharding_strategy: ttnn.TensorMemoryLayout = (
        ttnn.TensorMemoryLayout.WIDTH_SHARDED
    )
    mesh_mapper_config: MeshMapperConfig = field(
        default_factory=ReplicateMeshMapper
    )
```

For fusion artifacts, `metadata.json` includes `artifact_kind`:
`"fusion_group"` and a `views` object mapping logical names to serialized
view fields (`tensor_shape`, `shard_shape`, `core_range_set`, `dtype`,
`tile_shape`, `byte_offset`, `total_size`). `Fingerprint.target` is
`ArtifactTarget = TensorTarget | FusionGroupSpec`.

---

## Deployment and Storage Architecture

### Physical topology

Each host in the cluster has:

- **Local NVMe** — fast block storage attached to the host. Holds the
  per-host TensorCache (content-addressed artifacts). This is the hot
  path for model loading.
- **Shared NFS** — network-attached storage visible to all hosts. Holds
  the HuggingFace state dict (safetensors). This is the cold path, read
  only on cache misses.

```
┌──────────────────────────────────────────────────────────┐
│  Host N                                                  │
│                                                          │
│  ┌──────────────┐     ┌──────────────────────────────┐   │
│  │  NVMe SSD    │     │  Tenstorrent Devices (4×2)   │   │
│  │              │     │                              │   │
│  │  TensorCache │────►│  ttnn.load_tensor(data.tbin) │   │
│  │  objects/    │     │  → device L1 / DRAM           │   │
│  │    ab/...    │     │                              │   │
│  └──────────────┘     └──────────────────────────────┘   │
│         ▲                                                │
│         │ cache miss: preprocess + store                  │
│         │                                                │
│  ┌──────┴───────┐                                        │
│  │  NFS mount   │  ← /mnt/models/deepseek-ai/...        │
│  │  (HF state   │     safetensors (read-only)            │
│  │   dict)      │                                        │
│  └──────────────┘                                        │
└──────────────────────────────────────────────────────────┘
```

On a **warm start** (cache hit), each host reads only from its local
NVMe — no NFS traffic. On a **cold start** (first run or cache miss),
the host reads HF safetensors from NFS, preprocesses them, and writes
the artifacts to its local NVMe for future runs.

### Storage layout

```
NVMe (<local_nvme>/):
  objects/
    ab/
      cdef0123.../
        manifest.json       # Fingerprint (identity)
        metadata.json       # Content hash, size, timestamp
        data.tensorbin      # Tensor bytes (ttnn.dump_tensor format)
  tmp/
    {id}_{pid}/             # Atomic staging area (renamed on success)

NFS (<nfs_mount>/):
  deepseek-ai/DeepSeek-V3/
    model.safetensors.index.json
    model-00001-of-000NN.safetensors
    ...
```

### NVMe vs NFS performance

#### What we measured

All benchmarks are for loading **a single MoE layer** (~774 cache
files, 6.2 GB) on a single host with a 4×2 mesh. Each measurement
loads the full layer's cache from disk and sends every tensor to device
L1 or DRAM via `ttnn.load_tensor()`. All times are single-process (no
concurrent readers).

Three quantities were measured per run:

- **Cache generation time** — the first-time cost: read HF safetensors
  from NFS, run the preprocessing pipeline (transpose, quantize, fuse,
  tile), and write the cached tensorbin files to disk.
- **Cache load time** — the steady-state cost: read the cached
  tensorbin files from disk and send each tensor to device via
  `ttnn.load_tensor()`. This is what happens on every startup after
  the cache has been generated.
- **Raw read time** — read the same tensorbin files with plain
  `open()` + `read()` in 1 MB chunks, no device transfer, no mmap.
  Isolates storage throughput from everything else.

The OS **page cache** is the kernel's in-memory file cache (any
recently-read file stays in DRAM until evicted). To measure true
storage speed, we evict it before "cold" runs using `posix_fadvise
DONTNEED`. "Warm" runs pre-read the files so they are served from
host memory.

#### Headline result

| Scenario | NVMe | NFS | Ratio |
|----------|------|-----|-------|
| Cache load, cold page cache | **12.3 s** (510 MB/s) | **55.9 s** (113 MB/s) | 4.5× |
| Cache load, warm page cache | 12.3 s (510 MB/s) | 12.3 s (510 MB/s) | 1.0× |
| Raw read, cold page cache | 2.7 s (2,290 MB/s) | 13.2 s (475 MB/s) | 4.8× |
| Raw read, warm page cache | 0.5 s (~12 GB/s) | 0.6 s (~10 GB/s) | 1.2× |

Cold cache loads are what matter for production startup. NVMe cold
loads (12.3 s) are **4.5× faster** than NFS (55.9 s).

When the page cache is warm, both backends converge to the same
12.3 s — the **host-to-device PCIe ceiling** at ~510 MB/s. Storage
is irrelevant when files are already in DRAM.

#### Why the gap is larger than raw throughput explains

Loading a cached tensor is a two-stage pipeline: (1) read bytes from
storage, (2) transfer to device (L1 or DRAM). If the two stages overlap
well, end-to-end time is approximately `max(T_read, T_xfer)`.

```
Data:       D = 6,289 MB
PCIe xfer:  T_xfer = D / 510 MB/s = 12.3 s

             T_read    Best case       Actual     Pipelining
             (s)       max(T_r, T_x)   (s)
NVMe cold    2.7       12.3            12.3       perfect — read hides behind xfer
NFS cold    13.5       13.5            55.9       broken — 4.1× worse than expected
```

**NVMe cold loads match the device transfer ceiling.** The SSD reads
at 2.3 GB/s, fast enough that all data is in host memory before the
PCIe transfer finishes. The bottleneck is device transfer, not
storage.

**NFS cold loads are far worse than raw throughput predicts.** NFS
reads at 475 MB/s sequentially (raw read test), which would predict
a ~13.5 s load time. The actual 55.9 s is 4× worse. The cause is
how `ttnn.load_tensor()` accesses the data: it uses **mmap**, which
means:

1. Every first-access page triggers a **synchronous page fault**.
2. On NFS, each fault becomes a network round-trip (~0.5–5 ms).
3. The tensor deserialization access pattern is not purely sequential,
   so the NFS client's readahead is less effective than on a bulk
   `read()` call.
4. The high-latency faults **stall the device transfer pipeline** —
   the device sits idle waiting for the next page of data.

On NVMe, page faults are local I/O (~10–100 μs, 10–100× lower
latency). They complete fast enough that the device transfer pipeline
never stalls. This is why the raw throughput gap is 4.8× but the
cache load gap is 4.5× — NVMe doesn't just read faster, it pipelines
correctly.

#### Production implications

In the 64-stage pipeline, **4 stages share each host's NVMe and
memory**, and **16 hosts share the NFS server**. All stages load their
weights in parallel at startup. Wall-clock time is determined by the
slowest host, which is the host whose 4 concurrent readers finish
last.

**NVMe (4 readers per host):**

The single-reader benchmark shows NVMe cold loads are
device-transfer-bound at ~510 MB/s (the NVMe itself reads at
~2.3 GB/s). With 4 readers sharing the NVMe:

- NVMe bandwidth per reader: 2,300 / 4 = **575 MB/s** — still above
  the ~510 MB/s device transfer ceiling.
- Each reader remains device-transfer-bound, not storage-bound.
- Expected cold startup: still **~12 s** per host, assuming the 4
  device transfer paths don't contend with each other on the PCIe bus.

If the 4 meshes share PCIe bandwidth, the effective per-reader
transfer rate drops and startup time increases proportionally. This
is not yet measured.

**NFS (4 readers per host × 16 hosts = 64 readers):**

The single-reader benchmark already shows 55.9 s (mmap page fault
bottleneck). With 64 concurrent readers hitting the same NFS server:

- NFS server bandwidth is shared across all 64 readers.
- Per-reader NFS throughput drops well below the already-slow
  single-reader rate.
- The mmap page fault serialization problem compounds: more readers
  means more concurrent RPCs, longer NFS server queues, and higher
  per-fault latency.
- Expected cold startup: **significantly worse** than the 55.9 s
  single-reader measurement.

Neither the NVMe 4-reader contention nor the NFS 64-reader contention
is captured in the benchmarks above (all are single-process). Both are
worth measuring to validate the analysis.

#### Full benchmark data

The headline numbers above are medians from repeated runs. Full data
for reproducibility:

**Cold page cache** (page cache evicted before each load):

| Run | Storage | Cache gen (s) | Cache load (s) | Load MB/s | Raw read (s) | Raw read MB/s |
|-----|---------|---------------|----------------|-----------|--------------|---------------|
| 1   | NFS     | 244.7         | 52.42          | 120       | 12.70        | 495           |
| 2   | NFS     | 224.6         | 55.12          | 114       | 13.37        | 471           |
| 3   | NFS     | 224.0         | 60.12          | 105       | 12.64        | 498           |
| 4   | NVMe    | 210.9         | 12.32          | 511       | 2.72         | 2,311         |
| 5   | NVMe    | 209.7         | 17.33          | 363       | 2.73         | 2,308         |
| 6   | NVMe    | 209.9         | 12.35          | 509       | 2.71         | 2,324         |

**Warm page cache** (files pre-read into OS page cache):

| Run | Storage | Cache gen (s) | Cache load (s) | Load MB/s | Raw read, warm (s) | Raw read, cold (s) |
|-----|---------|---------------|----------------|-----------|---------------------|---------------------|
| 1   | NVMe    | 210.6         | 17.38          | 362       | 0.51                | 2.76                |
| 2   | NVMe    | 211.6         | 12.28          | 512       | 0.46                | 2.74                |
| 3   | NVMe    | 210.0         | 17.49          | 360       | 0.51                | 2.76                |
| 4   | NFS     | 223.6         | 12.33          | 510       | 0.58                | 14.45               |
| 5   | NFS     | 225.9         | 12.50          | 503       | 0.62                | 12.86               |
| 6   | NFS     | 226.0         | 17.53          | 359       | 0.63                | 13.20               |
| 7   | NFS     | 224.0         | 12.02          | 521       | 0.63                | 13.30               |

---

## FusionGroupSpec instances (reference)

### q_ab_kv_a (homogeneous bfloat8_b, WIDTH_SHARDED)

Fuses q_a_proj, q_b_proj, kv_a_proj into one buffer. q_a and q_b share
96 cores (q_ab region); kv_a occupies 18 cores (kv region), padded to
match the q_ab shard width. All shapes are **post-preprocessing**
(after transpose, head deinterleave, TP slice, and packing); they
describe the per-device layout in the fused buffer, not the raw HF
tensor shapes.

```python
Q_AB_KV_A_SPEC = FusionGroupSpec(
    name="q_ab_kv_a",
    regions=(
        RegionSpec(
            core_range_set=ttnn.CoreRangeSet([
                ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(11, 7)),
            ]),
            subtensors=(
                SubTensorSpec("q_a_proj",  (3584, 3072),  ttnn.bfloat8_b, (32, 32)),
                SubTensorSpec("q_b_proj",  (1536, 12288), ttnn.bfloat8_b, (32, 32)),
            ),
        ),
        RegionSpec(
            core_range_set=ttnn.CoreRangeSet([
                ttnn.CoreRange(ttnn.CoreCoord(0, 8), ttnn.CoreCoord(8, 9)),
            ]),
            subtensors=(
                SubTensorSpec("kv_a_proj", (7168, 576), ttnn.bfloat8_b, (32, 32)),
            ),
        ),
    ),
    sharding_strategy=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
)
```

### o_proj_gate_mm_norms (mixed dtype, WIDTH_SHARDED)

Fuses o_proj (bfp8), gate_mm (bfp16), and four RMSNorm gammas (bfp16,
1x32 tiles) into a UINT32 raw-byte buffer. Each core is padded to the
same max shard byte size.

```python
O_PROJ_GATE_MM_NORMS_SPEC = FusionGroupSpec(
    name="o_proj_gate_mm_norms",
    regions=(
        RegionSpec(
            core_range_set=ttnn.CoreRangeSet([
                ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(11, 7)),
                ttnn.CoreRange(ttnn.CoreCoord(1, 8), ttnn.CoreCoord(8, 9)),
            ]),
            subtensors=(
                SubTensorSpec("o_proj", (8192, 7168), ttnn.bfloat8_b, (32, 32)),
            ),
        ),
        RegionSpec(
            core_range_set=ttnn.CoreRangeSet([
                ttnn.CoreRange(ttnn.CoreCoord(12, 0), ttnn.CoreCoord(12, 7)),
            ]),
            subtensors=(
                SubTensorSpec("gate_mm", (7168, 256), ttnn.bfloat16, (32, 32)),
            ),
        ),
        RegionSpec(
            core_range_set=ttnn.CoreRangeSet([
                ttnn.CoreRange(ttnn.CoreCoord(12, 9), ttnn.CoreCoord(12, 9)),
            ]),
            subtensors=(
                SubTensorSpec("attn_norm", (1, 7168), ttnn.bfloat16, (1, 32)),
                SubTensorSpec("q_norm",    (1, 1536), ttnn.bfloat16, (1, 32)),
                SubTensorSpec("ffn_norm",  (1, 7168), ttnn.bfloat16, (1, 32)),
            ),
        ),
        RegionSpec(
            core_range_set=ttnn.CoreRangeSet([
                ttnn.CoreRange(ttnn.CoreCoord(0, 8), ttnn.CoreCoord(0, 8)),
            ]),
            subtensors=(
                SubTensorSpec("kv_norm", (1, 512), ttnn.bfloat16, (1, 32)),
            ),
        ),
    ),
    sharding_strategy=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
)
```

### kv_b12 (homogeneous bfloat8_b, HEIGHT_SHARDED)

Fuses kv_b1_proj and kv_b2_proj. Each occupies 64 cores in disjoint
ranges; kv_b2 is tile-shuffled to match the common shard shape.

```python
KV_B12_SPEC = FusionGroupSpec(
    name="kv_b12",
    regions=(
        RegionSpec(
            core_range_set=ttnn.CoreRangeSet([
                ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7)),
            ]),
            subtensors=(
                SubTensorSpec("kv_b1_proj", (8192, 512), ttnn.bfloat8_b, (32, 32)),
            ),
        ),
        RegionSpec(
            core_range_set=ttnn.CoreRangeSet([
                ttnn.CoreRange(ttnn.CoreCoord(8, 0), ttnn.CoreCoord(12, 7)),   # 5×8 = 40 cores
                ttnn.CoreRange(ttnn.CoreCoord(0, 8), ttnn.CoreCoord(11, 9)),   # 12×2 = 24 cores
            ]),
            subtensors=(
                SubTensorSpec("kv_b2_proj", (512, 8192), ttnn.bfloat8_b, (32, 32)),
            ),
        ),
    ),
    sharding_strategy=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
)
```

### gate_up (homogeneous bfloat4_b, HEIGHT_SHARDED)

Fuses shared gate_proj and up_proj. Gate occupies 64 A-compute cores,
up occupies 64 B-compute cores. Both reshuffled from block to
height-sharded layout.

```python
GATE_UP_SPEC = FusionGroupSpec(
    name="gate_up",
    regions=(
        RegionSpec(
            core_range_set=...,  # 64 A-compute cores
            subtensors=(
                SubTensorSpec("shared_gate_proj", (7168, 256), ttnn.bfloat4_b, (32, 32)),
            ),
        ),
        RegionSpec(
            core_range_set=...,  # 64 B-compute cores
            subtensors=(
                SubTensorSpec("shared_up_proj", (7168, 256), ttnn.bfloat4_b, (32, 32)),
            ),
        ),
    ),
    sharding_strategy=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
)
```

---

## `create_overlapped_tensor` (implemented, delegates to BlitzDecodeWeights)

`weights/fusion_runtime.py` defines `create_overlapped_tensor(spec,
preprocessed, device, ...) -> (fused_tensor, views)`. For the four
production DeepSeek fusion groups (`q_ab_kv_a`, `o_proj_gate_mm_norms`,
`kv_b12`, `gate_up`), it delegates to the existing `BlitzDecodeWeights`
`get_tt_*` methods so layout stays identical to the non-cache path.
`_validate_views_match_spec` asserts produced `OverlappedTensor` views
match `FusionGroupSpec` regions for drift detection.

A fully generic packer that replaces all `get_tt_*` logic is **not**
implemented; the module is the integration layer between CAS and the
current fusing code.

---

## Cache Interface

> **Implementation status:** `TensorCache` and `EphemeralTensorCache` in
> `tensor_cache/cache.py` implement `get_or_create` for both
> `TensorTarget` (returns `ttnn.Tensor`) and `FusionGroupSpec` (returns
> `dict[str, OverlappedTensor]`).

```python
class TensorCache:
    """Content-addressed cache backed by a local filesystem (NVMe).

    Storage layout:
        {local_root}/
          objects/{id[:2]}/{id}/
            manifest.json
            metadata.json
            data.tensorbin

    Writes go directly into the object directory; atomic publish via a
    staging directory is still TODO for crash/concurrency hardening.
    """

    def __init__(self, local_root: Path):
        self.local_root = Path(local_root)

    def get_or_create(
        self,
        fingerprint: Fingerprint,
        device,
        *,
        preprocess: Callable[
            [dict[str, torch.Tensor]], dict[str, torch.Tensor]
        ],
        raw_tensors: (
            Callable[[], dict[str, torch.Tensor]]
            | dict[str, torch.Tensor]
        ),
    ) -> ttnn.Tensor:
        """Return a device tensor, loading from cache on hit or
        building on miss.

        On hit:  load from local NVMe directly to device.
        On miss: raw_tensors() -> preprocess() -> from_torch(host)
                 -> store -> load to device.

        raw_tensors can be a callable (lazy) so HF safetensors are
        never read when the cache is warm.

        The mesh_mapper is reconstructed from target.mesh_mapper_config
        at runtime — it is not a separate parameter, ensuring it is
        always part of the fingerprint hash.
        """
        target = fingerprint.target
        artifact_id = compute_artifact_id(fingerprint)
        entry = self._lookup(artifact_id)

        if isinstance(entry, PresentCacheEntry):
            return self._load(entry.paths, device)

        if isinstance(entry, CorruptCacheEntry):
            shutil.rmtree(entry.paths.object_dir, ignore_errors=True)

        tensors = raw_tensors() if callable(raw_tensors) else raw_tensors
        preprocessed = preprocess(tensors)
        torch_tensor = preprocessed[target.name]

        from_torch_kwargs = {
            "dtype": target.dtype,
            "layout": target.layout,
            "device": None,
            "memory_config": target.memory_config,
            "mesh_mapper": self._build_mesh_mapper(target, device),
        }
        if target.layout == ttnn.TILE_LAYOUT:
            from_torch_kwargs["tile"] = ttnn.Tile(target.tile_shape)

        tensor_host = ttnn.from_torch(torch_tensor, **from_torch_kwargs)
        paths = self._store(artifact_id, fingerprint, tensor_host)
        return self._load(paths, device)
```

Key design decisions in the current implementation:

- **No `mesh_mapper` parameter.** The mesh mapping strategy is part of
  `TensorTarget.mesh_mapper_config`, which participates in the
  fingerprint hash. `TensorCache._build_mesh_mapper()` reconstructs the
  runtime `ttnn.MeshMapper` object from the declarative config and the
  device at call time. This prevents stale data from mesh mapper
  changes that would otherwise go undetected.

- **`tile` is conditional on layout.** For `ROW_MAJOR_LAYOUT` tensors
  (e.g. embedding), the `tile` kwarg is omitted from `from_torch` since
  it is only valid for `TILE_LAYOUT`.

- **Atomic writes.** Artifacts are staged in a `tmp/{id}_{pid}/`
  directory and atomically renamed to `objects/{id[:2]}/{id}/` on
  success. If another process wrote the same artifact concurrently, the
  rename fails harmlessly (same fingerprint guarantees same content).

---

## Fingerprint Canonicalization

> **Implementation status:** Implemented in `tensor_cache/fingerprint.py`
> for `TensorTarget`. `FusionGroupSpec` canonicalization will be added
> when fusion group support is implemented.

```python
def canonical(fingerprint: Fingerprint) -> dict:
    """Produce a deterministic, JSON-serializable dict from a Fingerprint."""
    target = fingerprint.target
    target_dict = {
        "kind": "tensor",
        "name": target.name,
        "dtype": target.dtype.name,
        "layout": target.layout.name,
        "memory_config": json.loads(target.memory_config.to_json()),
        "tile_shape": list(target.tile_shape),
        "mesh_mapper_config": _canonical_mesh_mapper(target.mesh_mapper_config),
    }
    return {
        "schema_version": fingerprint.schema_version,
        "source": sorted(fingerprint.source.names),
        "hf_model_id": fingerprint.hf_model_id,
        "hf_revision": fingerprint.hf_revision,
        "transform_version": fingerprint.transform_version,
        "mesh_shape": list(fingerprint.mesh_shape),
        "target": target_dict,
    }


def compute_artifact_id(fingerprint: Fingerprint) -> str:
    """SHA-256 hex digest of the canonical JSON representation."""
    blob = json.dumps(canonical(fingerprint), sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()
```

The `mesh_mapper_config` is included in the canonical representation
because the mesh mapping strategy affects the serialized tensor data.
A change from `replicate` to `shard(dim=1)` produces a different
artifact ID, preventing stale data.

tt-metal provides deterministic serialization for all heavy types:
`CoreRangeSet.to_json()`, `MemoryConfig.to_json()`, `DataType.name`,
`Layout.name`. No custom serialization is needed.

---

## Read Path (cache hit)

```
artifact_id = sha256(canonical(fingerprint))
entry = lookup(artifact_id)
    if ABSENT:  → fall through to write path
    if CORRUPT: → delete entry, fall through to write path
    if PRESENT: → load and return

tensor = ttnn.load_tensor(entry.paths.data_path, device)
return tensor
```

For standalone tensors (`TensorTarget`), the hit path is a single
`ttnn.load_tensor` call — no view reconstruction. For fusion groups
(`FusionGroupSpec`), the hit path loads the fused `data.tensorbin` and
reconstructs `OverlappedTensor` views from the `views` field in
`metadata.json` (see `tensor_cache/overlapped_metadata.py`).

## Write Path (cache miss)

```
tensors = raw_tensors()                    # lazy: only called on miss
preprocessed = preprocess(tensors)         # model-specific torch transforms

# TensorTarget: build host tensor, dump, write manifest + metadata, load to device
# FusionGroupSpec: create_overlapped_tensor → dump fused host tensor,
#   write manifest + metadata (including views), load fused tensor + rebuild views

# Current implementation writes directly under objects/{id[:2]}/{id}/;
# atomic staging + rename is TODO (see TensorCache._write_artifact_blob_and_manifest).
```

---

## Model-Level Usage

> **Implementation status:** `weights/adapter.py` (importable as `prepare_weights`) routes **all** weight
> artifacts (standalone tensors, fusion groups, per-expert routed
> weights) through `cache_config.cache.get_or_create(...)`. When
> `cache_config` is omitted, `prepare_*` uses
> `CacheConfig.ephemeral()` (`EphemeralTensorCache`) so there is a
> single code path with no disk persistence.

A `CURRENT_TRANSFORM_VERSION` constant in `weights/catalog.py` (re-exported from `prepare_weights`) is bumped
when preprocessing logic changes, invalidating cached artifacts.

```python
# weights/adapter.py + catalog (conceptual; `prepare_weights` is a facade)

CURRENT_TRANSFORM_VERSION = 1

def prepare_embedding_weights(..., cache_config: CacheConfig | None = None):
    if cache_config is None:
        cache_config = CacheConfig.ephemeral(move_to_device=move_to_device)
    # ... fingerprint + get_or_create for embedding TensorTarget ...
```

`CacheContext` bundles the per-model fields so they are not repeated at
every `get_or_create` call site. `CacheConfig` pairs it with the
`TensorCache` instance:

```python
from models.demos.deepseek_v3_b1.tensor_cache import CacheConfig, CacheContext, TensorCache

cache_config = CacheConfig(
    cache=TensorCache(Path("/local_nvme/deepseek_v3_cache")),
    context=CacheContext(
        schema_version=1,
        hf_model_id="deepseek-ai/DeepSeek-V3",
        hf_revision="d1a891dd58e6bb0a671bfc6f3046e29e3478e924",
        transform_version=CURRENT_TRANSFORM_VERSION,
        mesh_shape=(4, 2),
    ),
)

# Each prepare function uses cache_config internally:
embedding = prepare_embedding_weights(state_dict, device, cache_config=cache_config)
lm_head = prepare_lm_head_weights(state_dict, device, cache_config=cache_config)
mtp = prepare_mtp_weights(state_dict, device, cache_config=cache_config)
```

Seven standalone tensors are currently cached through this mechanism:

| Tensor | Prepare function | `TensorTarget` key fields |
|--------|-----------------|--------------------------|
| embedding | `prepare_embedding_weights` | ROW_MAJOR, DRAM, replicate |
| lm_head | `prepare_lm_head_weights` | bfp8, WIDTH_SHARDED L1 (101 cores), shard(dim=1) |
| final_norm | `prepare_lm_head_weights` | bf16, HEIGHT_SHARDED L1 (mcast core), tile 1x32 |
| gate_bias | `prepare_attention_weights` | bf16, HEIGHT_SHARDED L1 (sender core), tile 16x16 |
| mtp_h_gamma | `prepare_mtp_weights` | bf16, HEIGHT_SHARDED L1 (mcast core), tile 1x32 |
| mtp_e_gamma | `prepare_mtp_weights` | bf16, HEIGHT_SHARDED L1 (mcast core), tile 1x32 |
| mtp_eh_projection | `prepare_mtp_weights` | bfp8, WIDTH_SHARDED DRAM (8 banks) |

Many other weights use `TensorTarget` fingerprints as well (e.g.
`shared_down_proj`, dense stacked routed projections, and one cache entry
per MoE routed expert × projection).

### Fusion groups and routed experts

Fusion groups (`Q_AB_KV_A_SPEC`, `O_PROJ_GATE_MM_NORMS_SPEC`, `KV_B12_SPEC`,
`GATE_UP_SPEC`) and routed/shared weights use the same `get_or_create`
pattern with `FusionGroupSpec` or `TensorTarget` fingerprints. MoE
routed experts use one `TensorTarget` shape per projection type and
768 cache entries per layer (256× gate/up/down). `generate_cache.py`
warms the CAS root by calling `prepare_*` with a disk-backed
`CacheConfig`; `CacheWeightProvider` can populate on miss at runtime.

---

## Current state and layout

### Unified design (today)

| File | Role |
|---|---|
| `blitz_decode_weights.py` | Overlap specs, `FusionGroupSpec` constants (`Q_AB_KV_A_SPEC`, …), `BlitzDecodeWeights` `get_tt_*` fusing, `OverlappedTensor` |
| `prepare_weights.py` | Compatibility facade re-exporting `weights/` |
| `weights/` | `catalog` (artifact targets), `preprocessing`, `fusion_runtime` (pack/fuse), `types` (assembled dataclasses), `adapter` (`prepare_*` orchestration) |
| `tensor_cache/` | Types, fingerprinting, `TensorCache` / `EphemeralTensorCache`, `fuse.py` (shim to `weights/fusion_runtime`), `overlapped_metadata.py` |
| `scripts/generate_cache.py` | Offline CAS warm; `--verify` double-load via `CacheWeightProvider` |
| `demo/weight_provider.py` | `CacheWeightProvider`, `SyntheticWeightProvider`, `StateDictWeightProvider` |

**Residual limitations** (see also “Known limitations” below):

- **Duplicated fusing implementations:** `weights/fusion_runtime.py` (via `tensor_cache/fuse.py` shim) delegates to
  `get_tt_*`; there is no single generic packer for all four groups.
- **Non-atomic CAS writes:** objects are written in place under
  `objects/`; concurrent writers or interrupted writes need a future
  staging/rename hardening.
- **Content hash:** stored in `metadata.json` but not verified on every
  load by default.

### Overlap spec classes → `FusionGroupSpec` (implemented)

| Overlap spec class | Spec constant | Notes |
|---|---|---|
| `QAB_KVA_PROJ_SingleDeviceOverlapSpec` | `Q_AB_KV_A_SPEC` | Fingerprint + cache target; preprocess in `prepare_attention_weights` |
| `O_PROJ_GATE_MM_RMSNORM_GAMMA_SingleDeviceOverlapSpec` | `O_PROJ_GATE_MM_NORMS_SPEC` | Dense layers use zero `gate_mm` with same spec for layout |
| `KVB12_PROJ_SingleDeviceOverlapSpec` | `KV_B12_SPEC` | |
| `GATE_UP_PROJ_SingleDeviceOverlapSpec` | `GATE_UP_SPEC` | |
| `DOWN_PROJ_SingleDeviceSpec` | `TensorTarget` (`shared_down_proj`) | Not a fusion group |

### Disk persistence

| Mechanism | Role |
|---|---|
| `prepare_*` + `CacheConfig(TensorCache, …)` | On miss, stores under `objects/{id[:2]}/{id}/` |
| `CacheWeightProvider` | Same `prepare_*` path; CAS hits avoid HF reads when `raw_tensors` is lazy |
| `scripts/generate_cache.py` | Offline warm; legacy `layer_NNN/manifest.json` trees are not supported |

### Migration path

TensorCache (`objects/`) is the only supported durable layout for this
demo tree. Regenerate or drop any on-disk `layer_NNN/` caches that
predate CAS.

### What does NOT change

- **`OverlappedTensor` dataclass** — remains the runtime view type used
  by kernels. The CAS produces these on cache hit (from stored
  metadata) and on cache miss (from `create_overlapped_tensor`).

- **`DeepSeekV3MoELayerWeights` / `DeepSeekV3DenseLayerWeights`** —
  remain the model-level weight containers. The cache produces their
  fields; the dataclass structure is unchanged.

- **`WeightProvider` protocol** — unchanged interface; providers call
  the same `prepare_*` entry points with or without a disk-backed cache.

- **Kernel code** — completely untouched. Kernels consume
  `OverlappedTensor` views; they don't know or care how those views
  were produced or cached.

- **Preprocessing logic** — lives in `weights/preprocessing.py` and inline preprocess lambdas in `weights/adapter.py`
  closures and in `BlitzDecodeWeights` / `get_tt_*`; behavior is unchanged
  from the pre-CAS layout aside from routing through `get_or_create`.

### Known limitations and future work

- **`transform_version` is a manual integer.** If a preprocessing
  function changes without bumping its `transform_version`, the cache
  serves stale data — the same class of bug the old manifest `version`
  had. Content addressing eliminates most staleness vectors (source
  tensors, mesh shape, target layout are all hashed), but the
  preprocessing code itself is only versioned, not hashed. Future
  mitigations to explore:
  1. Hash the preprocessing function's source code automatically.
  2. Annotate each preprocessing function with a version decorator
     that the fingerprint reads, reducing the chance of forgetting.
  3. Add a CI check that compares output hashes of a reference input
     across versions, catching silent regressions.

- **Cache eviction is manual.** At 362 GB per model version, NVMe
  space management matters. For now, stale artifacts (e.g. from old
  `transform_version`s or old model revisions) are removed manually.
  Future work should add an eviction strategy — for example, removing
  artifacts whose `transform_version` is older than the current code,
  or evicting by least-recently-used / oldest-unused.

- **No concurrency protection for same-artifact cache misses.**
  Pipeline stages are not expected to request the same artifact
  simultaneously (each stage loads its own layer's weights), so
  duplicate preprocessing work is not a concern today. If this
  assumption changes (e.g. shared embedding loaded by multiple
  stages), a file-lock mechanism should be added to avoid redundant
  NFS reads and preprocessing.
