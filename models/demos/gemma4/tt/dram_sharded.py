# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Peak-DRAM-bandwidth decode weight matmuls (Phase 2a).

Single-user decode is weight-bandwidth bound: each weight byte is read once per
token. A plain ``ttnn.linear`` over a DRAM-*interleaved* weight leaves DRAM
bandwidth on the table; a ``MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig``
matmul over a DRAM-*width-sharded* weight streams the weight from every DRAM bank
at peak bandwidth. This module packages that pattern so attention (qkv / o_proj)
and the lm_head can reuse the exact recipe the SharedMLP validated (~+7% tok/s):

  build once (at weight load):  DramShardedMatmul(mesh_device, interleaved_w, k, n)
  call in decode:               out = mm(x)   # x DRAM-interleaved -> out DRAM-interleaved

Design choices that keep call sites drop-in and safe:
- The weight is stored DRAM-*interleaved* by the caller as usual (so the *prefill*
  matmul is byte-for-byte unchanged — a plain 2D matmul cannot consume a
  DRAM-width-sharded in1: it asserts "Only L1 buffers can have an associated
  circular buffer"). We hold a *separate* persistent DRAM-width-sharded copy that
  only the decode path reads (costs ~1x that weight's DRAM while enabled).
- The activation is resharded to L1 width-sharded on entry and the output is
  converted back to DRAM-interleaved on exit, so head-split / concat / all-reduce
  / softcap all see the same layouts they did before.
- ``try_build`` returns ``None`` if the dims don't shard cleanly or the sharded
  copy can't be allocated (OOM), so enabling the flag can never crash a run.
"""

import math
import os

from loguru import logger

import ttnn

TILE = 32


def env_flag(*names, default=False):
    """True if any of the given env vars is set truthy (1/true/yes/on)."""
    for name in names:
        v = os.environ.get(name)
        if v is not None and v.strip().lower() in ("1", "true", "yes", "on"):
            return True
    return default


def _find_grid_k_n(k_tiles, n_tiles, max_rows=8, max_cols=8):
    """Largest core-grid (rows, cols) whose core count divides both k_tiles and n_tiles."""
    max_cores = max_rows * max_cols
    possible = [c for c in range(1, max_cores + 1) if k_tiles % c == 0 and n_tiles % c == 0]
    possible.sort(reverse=True)
    for cores in possible:
        for rows in range(1, max_rows + 1):
            if cores % rows == 0:
                cols = cores // rows
                if cols <= max_cols:
                    return rows, cols
    raise AssertionError(f"No core grid divides both k_tiles={k_tiles} and n_tiles={n_tiles}")


def find_k_core_grid(k, max_rows=8, max_cols=8):
    """Largest (rows, cols) such that ``k % (TILE * rows * cols) == 0``.

    Used for lm_head multi-split: core count is driven by K (hidden), not
    ``gcd(k_tiles, n_tiles)``. For Gemma4 hidden=5376 this yields 7×8 = 56 cores.
    """
    assert k % TILE == 0, f"K {k} not tile-aligned"
    k_tiles = k // TILE
    max_cores = max_rows * max_cols
    for cores in range(max_cores, 0, -1):
        if k_tiles % cores != 0:
            continue
        for rows in range(1, max_rows + 1):
            if cores % rows == 0:
                cols = cores // rows
                if cols <= max_cols:
                    return rows, cols
    raise AssertionError(f"No core grid divides k_tiles={k_tiles}")


def plan_column_splits(n, max_columns):
    """Split ``n`` columns into chunks of at most ``max_columns`` (last may be smaller)."""
    if max_columns < TILE:
        raise ValueError(f"max_columns must be >= {TILE}, got {max_columns}")
    # Tile-align the split width so each shard is a valid matmul N.
    max_columns = (max_columns // TILE) * TILE
    if max_columns < TILE:
        max_columns = TILE
    num_splits = math.ceil(n / max_columns)
    sizes = [max_columns] * (num_splits - 1)
    sizes.append(n - sum(sizes))
    assert sum(sizes) == n and all(s > 0 for s in sizes)
    return sizes


def pad_n_for_cores(n, num_cores):
    """Round ``n`` up so ``n % (TILE * num_cores) == 0``.

    DRAM-sharded matmul uses ``per_core_N = ceil(n / (TILE * num_cores))``, which
    silently widens the output when N is not core-aligned. For lm_head splits
    (e.g. N=8192 on 56 cores → effective 8960) that padding must be real zeros
    in the weight, and the caller must crop logits back to the logical width
    before concat — otherwise vocab columns scramble and decode looks like garbage.
    """
    step = TILE * num_cores
    return math.ceil(n / step) * step


def _largest_divisor(n, max_divisor=8):
    for i in range(max_divisor, 0, -1):
        if n % i == 0:
            return i
    return 1


def _tile_bytes(dtype):
    """Approx L1 bytes for one 32x32 tile at the given dtype (incl. bfp exponents)."""
    return {
        ttnn.bfloat16: 2048,
        ttnn.bfloat8_b: 1088,  # 1024 data + 64 block-exp
        ttnn.bfloat4_b: 576,  # 512 data + 64 block-exp
    }.get(dtype, 2048)


# Usable L1 for statically-allocated circular buffers (Blackhole ~1.5 MB total;
# leave headroom for the ops we don't model — semaphores, small scratch CBs).
_L1_CB_BUDGET_BYTES = 1_300_000


def _dram_weight_mem_config(k, n, dram_cores):
    padded_n = math.ceil(n / (TILE * dram_cores)) * (TILE * dram_cores)
    dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_cores - 1, 0))})
    shard_spec = ttnn.ShardSpec(dram_grid, (k, padded_n // dram_cores), ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard_spec)


class DramShardedMatmul:
    """A decode-mode DRAM-width-sharded weight matmul (see module docstring)."""

    def __init__(
        self,
        mesh_device,
        interleaved_weight,
        k,
        n,
        m=TILE,
        fused_activation=None,
        name="",
        num_cores=None,
        weight_already_dram_sharded=False,
        math_fidelity=None,
    ):
        assert mesh_device.dram_grid_size().y == 1, "DRAM sharding assumes a single DRAM row"
        dram_cores = mesh_device.dram_grid_size().x
        assert k % TILE == 0 and n % TILE == 0, f"{name}: dims ({k},{n}) not tile-aligned"

        if num_cores is not None:
            # K-driven grid (lm_head multi-split). Find a (rows, cols) factorization
            # that matches the requested core count and divides K.
            assert k % (TILE * num_cores) == 0, f"{name}: K {k} not divisible by tile*num_cores {TILE * num_cores}"
            rows, cols = None, None
            for r in range(1, 9):
                if num_cores % r == 0:
                    c = num_cores // r
                    if c <= 8:
                        rows, cols = r, c
                        break
            assert rows is not None, f"{name}: cannot factor num_cores={num_cores} into ≤8×8 grid"
        else:
            rows, cols = _find_grid_k_n(k // TILE, n // TILE)
        self.num_cores = rows * cols
        assert k % (TILE * self.num_cores) == 0, f"{name}: K {k} not divisible by tile*cores {TILE*self.num_cores}"

        in0_block_w = _largest_divisor(k // (TILE * self.num_cores))
        per_core_M = math.ceil(m / TILE)
        per_core_N = math.ceil(n / (TILE * self.num_cores))

        # Guard against L1 circular-buffer overflow BEFORE touching the device. When
        # gcd(k_tiles, n_tiles) is small (e.g. lm_head: k=5376, n=65536 -> 8 cores),
        # per_core_N is huge and the weight CB alone blows past L1 (lm_head needs
        # ~11.7 MB vs 1.5 MB). Estimate the dominant CBs and raise so try_build
        # falls back to the interleaved matmul instead of crashing at runtime.
        wtb = _tile_bytes(interleaved_weight.dtype)
        in1_cb = 2 * in0_block_w * per_core_N * wtb  # weight, double-buffered
        out_cb = 2 * per_core_M * per_core_N * 2048  # bf16 output
        in0_cb = 2 * per_core_M * in0_block_w * 2048  # activation
        est_l1 = in1_cb + out_cb + in0_cb
        assert est_l1 <= _L1_CB_BUDGET_BYTES, (
            f"{name}: estimated L1 CBs {est_l1} B > budget {_L1_CB_BUDGET_BYTES} B "
            f"(per_core_N={per_core_N}, cores={self.num_cores}); dims unsuited to DRAM-sharded matmul"
        )

        if weight_already_dram_sharded:
            # Caller loaded via as_tensor(..., memory_config=DRAM-width-sharded).
            # Do NOT to_memory_config again — a second convert can corrupt the
            # shard layout under mesh TP.
            self.weight = interleaved_weight
        else:
            self.weight = ttnn.to_memory_config(interleaved_weight, _dram_weight_mem_config(k, n, dram_cores))
        self._in_mem_config = ttnn.create_sharded_memory_config(
            (m, k // self.num_cores),
            ttnn.CoreGrid(y=rows, x=cols),
            ttnn.ShardStrategy.WIDTH,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self._prog_config = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=in0_block_w,
            per_core_M=per_core_M,
            per_core_N=per_core_N,
            fused_activation=fused_activation,
        )
        # Peer LMHead: HiFi2 + packer_l1_acc. Default LoFi → "donkeys" on bf16
        # lm_head. HiFi2 still flips greedy argmax vs interleaved ~6% of the
        # time on real weights (tiny top-2 margins). Override via env for A/B:
        #   GEMMA4_DRAM_MATH_FIDELITY=HiFi4|HiFi2|LoFi|none
        #   GEMMA4_DRAM_FP32_DEST_ACC=1
        # ``math_fidelity`` arg (from MultiSplit) wins over the env default.
        if math_fidelity is None:
            fidelity_name = os.environ.get("GEMMA4_DRAM_MATH_FIDELITY", "HiFi2").strip()
        else:
            fidelity_name = math_fidelity
        if fidelity_name.lower() in ("none", "off", "0", "default"):
            self._compute_kernel_config = None
        else:
            fidelity = getattr(ttnn.MathFidelity, fidelity_name, ttnn.MathFidelity.HiFi2)
            fp32_acc = env_flag("GEMMA4_DRAM_FP32_DEST_ACC")
            self._compute_kernel_config = ttnn.WormholeComputeKernelConfig(
                math_fidelity=fidelity,
                math_approx_mode=False,
                fp32_dest_acc_en=fp32_acc,
                packer_l1_acc=True,
            )
        self.k = k
        self.n = n
        self.name = name
        self.rows = rows
        self.cols = cols

    @classmethod
    def try_build(
        cls,
        mesh_device,
        interleaved_weight,
        k,
        n,
        m=TILE,
        fused_activation=None,
        name="",
        num_cores=None,
        weight_already_dram_sharded=False,
        math_fidelity=None,
    ):
        """Build the matmul, or return ``None`` (and log) if dims don't shard or alloc fails."""
        try:
            return cls(
                mesh_device,
                interleaved_weight,
                k,
                n,
                m=m,
                fused_activation=fused_activation,
                name=name,
                num_cores=num_cores,
                weight_already_dram_sharded=weight_already_dram_sharded,
                math_fidelity=math_fidelity,
            )
        except Exception as e:  # noqa: BLE001 - dims not shardable / OOM on the sharded copy
            logger.warning(f"DramShardedMatmul[{name}] disabled ({e}); using interleaved matmul")
            return None

    def __call__(self, x, out_memory_config=None, *, x_already_sharded=False):
        """x (DRAM-interleaved, or L1-sharded if ``x_already_sharded``) -> out DRAM-interleaved."""
        if x_already_sharded:
            x_sh = x
            own_x_sh = False
        else:
            x_sh = ttnn.to_memory_config(x, self._in_mem_config)
            own_x_sh = True
        linear_kwargs = dict(
            program_config=self._prog_config,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,  # match interleaved lm_head; avoid bfp8 pack noise
        )
        if self._compute_kernel_config is not None:
            linear_kwargs["compute_kernel_config"] = self._compute_kernel_config
        out = ttnn.linear(x_sh, self.weight, **linear_kwargs)
        if own_x_sh:
            x_sh.deallocate(True)
        out_i = ttnn.sharded_to_interleaved(out, out_memory_config or ttnn.DRAM_MEMORY_CONFIG)
        out.deallocate(True)
        return out_i

    def deallocate(self):
        if self.weight is not None:
            self.weight.deallocate(True)
            self.weight = None


class MultiSplitDramShardedMatmul:
    """Decode-mode multi-split DRAM-sharded matmul for fat N (lm_head).

    Slices the already-correct mesh-sharded interleaved lm_head weight along N
    (per-device vocab columns), then runs each chunk as a ``DramShardedMatmul``
    with a gcd(K,N) core grid so N is exact (no pad/crop). Prefill keeps the
    full interleaved weight.

    Host-side re-shard (cat per-device chunks + mesh_mapper) was tried and
    produced garbage under Gemma4's ``ShardTensor2dMesh`` TP — so we slice the
    validated mesh tensor instead.
    """

    def __init__(
        self,
        mesh_device,
        mesh_weight,
        k,
        n,
        max_columns=None,
        m=TILE,
        name="lm_head",
    ):
        """
        Args:
            mesh_weight: device mesh tensor ``[1,1,K,N_per_device]`` (already TP-sharded).
            n: vocab columns **per TP device**.
        """
        if max_columns is None:
            # Prefer a width that divides n exactly and lands on a large gcd grid.
            # 7168 → 56 cores; last remainder 1024 → 8 cores (both exact, no pad).
            max_columns = int(os.environ.get("GEMMA4_LM_HEAD_MAX_COLUMNS", "7168"))
        split_sizes = plan_column_splits(n, max_columns)
        assert mesh_weight.shape[-1] == n, f"{name}: mesh weight N {mesh_weight.shape[-1]} != expected per-device n {n}"

        # Peer LMHead uses one K-derived core grid for *every* split. Mixing
        # grids (e.g. 56-core for N=7168 + 8-core for N=1024) diverges in the
        # decode demo even when synthetic PCC looks fine. Pad N so every split
        # lands on the same grid, then crop logits back to the logical width.
        rows, cols = find_k_core_grid(k)
        num_cores = rows * cols

        self.splits = []  # (DramShardedMatmul|None, logical_n, interleaved_w|None)
        self._seeds = []  # keep slices/pads alive — to_memory_config may share storage
        offset = 0
        core_counts = []
        padded_ns = []
        use_interleaved = env_flag("GEMMA4_LM_HEAD_MULTI_SPLIT_INTERLEAVED")
        for i, split_n in enumerate(split_sizes):
            # Identity: single full-width split reuses the mesh weight (no slice).
            # Full-width ttnn.slice of lm_head has hung on this mesh; also avoids
            # an extra copy when max_columns >= n.
            if len(split_sizes) == 1 and split_n == n:
                w_slice = mesh_weight
                own_slice = False
            else:
                w_slice = ttnn.slice(
                    mesh_weight,
                    [0, 0, 0, offset],
                    [1, 1, k, offset + split_n],
                )
                own_slice = True
            if use_interleaved:
                if own_slice:
                    self._seeds.append(w_slice)
                self.splits.append((None, split_n, w_slice))
                core_counts.append(0)
                padded_ns.append(split_n)
                offset += split_n
                continue

            padded_n = pad_n_for_cores(split_n, num_cores)
            if padded_n > split_n:
                w_for_mm = ttnn.pad(
                    w_slice,
                    [(0, 0), (0, 0), (0, 0), (0, padded_n - split_n)],
                    value=0.0,
                )
                if own_slice:
                    self._seeds.append(w_slice)
                self._seeds.append(w_for_mm)
                own_slice = False  # already retained in _seeds
            else:
                w_for_mm = w_slice

            # HiFi4 cuts greedy argmax flips vs interleaved (0/32 vs ~6% at HiFi2)
            # on real lm_head weights. Still not byte-identical to default
            # interleaved (which itself uses a different CK), so keep opt-in.
            mm = DramShardedMatmul.try_build(
                mesh_device,
                w_for_mm,
                k=k,
                n=padded_n,
                m=m,
                name=f"{name}_split{i}",
                num_cores=num_cores,
                math_fidelity=os.environ.get("GEMMA4_DRAM_MATH_FIDELITY", "HiFi4"),
            )
            if mm is None:
                if own_slice:
                    w_slice.deallocate(True)
                for prev in self.splits:
                    if prev[0] is not None:
                        prev[0].deallocate()
                for seed in self._seeds:
                    seed.deallocate(True)
                raise RuntimeError(
                    f"{name}: split {i} (n={split_n}, padded={padded_n}) failed "
                    f"DramShardedMatmul build; falling back to interleaved"
                )
            if own_slice:
                self._seeds.append(w_slice)
            self.splits.append((mm, split_n, None))
            core_counts.append(mm.num_cores)
            padded_ns.append(padded_n)
            offset += split_n
        assert offset == n
        self.k = k
        self.n = n
        self.m = m
        self.name = name
        self.num_cores = num_cores
        self._interleaved_mode = use_interleaved
        logger.info(
            f"MultiSplitDramShardedMatmul[{name}]: {len(self.splits)} splits "
            f"(sizes={split_sizes}, padded={padded_ns}, cores={core_counts}, "
            f"interleaved={use_interleaved}) mesh-sliced unified-k-grid={rows}x{cols}"
        )

    @classmethod
    def try_build(
        cls,
        mesh_device,
        mesh_weight,
        k,
        n,
        max_columns=None,
        m=TILE,
        name="lm_head",
    ):
        try:
            return cls(
                mesh_device,
                mesh_weight,
                k,
                n,
                max_columns=max_columns,
                m=m,
                name=name,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"MultiSplitDramShardedMatmul[{name}] disabled ({e}); using interleaved matmul")
            return None

    def __call__(self, x, out_memory_config=None):
        outputs = []
        if self._interleaved_mode:
            # Optional: match DramShardedMatmul's HiFi2 CK so we can A/B whether
            # demo drift is fidelity vs DRAM-shard layout.
            fid = os.environ.get("GEMMA4_LM_HEAD_INTERLEAVED_FIDELITY", "").strip()
            if not fid and env_flag("GEMMA4_LM_HEAD_INTERLEAVED_HIFI2"):
                fid = "HiFi2"
            ck = None
            if fid:
                ck = ttnn.WormholeComputeKernelConfig(
                    math_fidelity=getattr(ttnn.MathFidelity, fid),
                    math_approx_mode=False,
                    fp32_dest_acc_en=False,
                    packer_l1_acc=True,
                )
            for _mm, _n, w in self.splits:
                if ck is not None:
                    outputs.append(ttnn.linear(x, w, compute_kernel_config=ck))
                else:
                    outputs.append(ttnn.linear(x, w))
        else:
            # All splits share one K-grid, so peer-style reuse of one L1-sharded
            # activation is safe (and cheaper than resharding 10×).
            x_sh = None
            for i, (mm, logical_n, _w) in enumerate(self.splits):
                if x_sh is None:
                    x_sh = ttnn.to_memory_config(x, mm._in_mem_config)
                out_i = mm(x_sh, out_memory_config=ttnn.DRAM_MEMORY_CONFIG, x_already_sharded=True)
                if out_i.shape[-1] > logical_n:
                    # Crop pad columns introduced for core alignment.
                    s = out_i.shape
                    cropped = ttnn.slice(
                        out_i,
                        [0, 0, 0, 0],
                        [s[0], s[1], s[2], logical_n],
                    )
                    out_i.deallocate(True)
                    out_i = cropped
                outputs.append(out_i)
            if x_sh is not None:
                x_sh.deallocate(True)
        if len(outputs) == 1:
            # concat of a single tensor can alias the input; do not deallocate.
            return outputs[0]
        out = ttnn.concat(outputs, dim=-1, memory_config=out_memory_config or ttnn.DRAM_MEMORY_CONFIG)
        # Do not deallocate ``outputs`` here. concat may share storage with an
        # input; freeing them corrupted logits and produced textsf loops.
        return out

    def deallocate(self):
        for entry in self.splits:
            mm = entry[0]
            if mm is not None:
                mm.deallocate()
        self.splits = []
        for seed in self._seeds:
            seed.deallocate(True)
        self._seeds = []
