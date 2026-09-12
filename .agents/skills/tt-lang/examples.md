## Reference Examples

### Example 1: Basic Pipe Send/Recv

Simplest pipe pattern. Core 0 loads a tile from DRAM and sends it to core 1 via pipe. Core 1 receives and writes to DRAM. Shows `PipeNet`, `if_src`, `if_dst` API. Note: pipes require a fixed grid size (not `grid="auto"`).

```python
@ttl.kernel(grid=(2, 1))
def pipe_send_recv(inp, out):
    net = ttl.PipeNet([ttl.Pipe((0, 0), (1, 0))])

    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        x, _ = ttl.node(dims=2)
        if x == 1:
            with inp_dfb.wait() as blk, out_dfb.reserve() as o:
                o.store(blk)

    @ttl.datamovement()
    def dm_read():
        x, _ = ttl.node(dims=2)
        if x == 0:
            with inp_dfb.reserve() as blk:
                tx = ttl.copy(inp[0, 0], blk); tx.wait()
                def send(pipe):
                    xf = ttl.copy(blk, pipe); xf.wait()
                net.if_src(send)
        if x == 1:
            with inp_dfb.reserve() as blk:
                def recv(pipe):
                    xf = ttl.copy(pipe, blk); xf.wait()
                net.if_dst(recv)

    @ttl.datamovement()
    def dm_write():
        x, _ = ttl.node(dims=2)
        if x == 1:
            with out_dfb.wait() as blk:
                tx = ttl.copy(blk, out[0, 0]); tx.wait()
```

From `test_pipe_basic.py`. Key points:
- Send happens inside `dm_read` after loading data, inside the same `reserve` block
- Receive happens in `dm_read` on the destination core
- Core 0 has no compute (it only sends); core 1 has no DRAM read (it only receives)

### Example 2: Ring Pipe (Neighbor Exchange)

Each core loads its own tile, sends it to the next core via a ring, receives its neighbor's tile, and adds them. This is the neighbor-sharing pattern used in molecular dynamics.

```python
N_CORES = 4

@ttl.kernel(grid=(N_CORES, 1))
def pipe_ring(inp, out):
    net = ttl.PipeNet([
        ttl.Pipe((x, 0), ((x + 1) % N_CORES, 0))
        for x in range(N_CORES)
    ])

    own_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), buffer_factor=2)
    nbr_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        with own_dfb.wait() as own, nbr_dfb.wait() as nbr, out_dfb.reserve() as o:
            o.store(own + nbr)

    @ttl.datamovement()
    def dm_read():
        x, _ = ttl.node(dims=2)
        with own_dfb.reserve() as blk:
            tx = ttl.copy(inp[0, x], blk); tx.wait()
            def send(pipe):
                xf = ttl.copy(blk, pipe); xf.wait()
            net.if_src(send)
        with nbr_dfb.reserve() as blk:
            def recv(pipe):
                xf = ttl.copy(pipe, blk); xf.wait()
            net.if_dst(recv)

    @ttl.datamovement()
    def dm_write():
        x, _ = ttl.node(dims=2)
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[0, x]); tx.wait()
```

From `test_pipe_ring.py`. Key points:
- Ring topology via `(x+1) % N_CORES` wraparound
- Every core both sends and receives (symmetric)
- `if_src`/`if_dst` dispatch to the correct pipe automatically per core

### Example 3: Scaled Dot-Product Attention (SDPA)

Single-core attention kernel showing the full softmax decomposition: Q@K^T, scale, row-wise max, shift, exp, sum, divide, then attn@V. Single-core is fine for getting the initial pattern working before scaling to multicore with streaming.

```python
SEQ_TILES = 1   # 32 tokens
HEAD_TILES = 2  # 64-dim head

@ttl.kernel(grid=(1, 1))
def sdpa_kernel(Q, K, V, scale, scaler, out):
    q_dfb = ttl.make_dataflow_buffer_like(Q, shape=(SEQ_TILES, HEAD_TILES), buffer_factor=1)
    k_dfb = ttl.make_dataflow_buffer_like(K, shape=(SEQ_TILES, HEAD_TILES), buffer_factor=1)
    v_dfb = ttl.make_dataflow_buffer_like(V, shape=(SEQ_TILES, HEAD_TILES), buffer_factor=1)
    scale_dfb = ttl.make_dataflow_buffer_like(scale, shape=(1, 1), buffer_factor=1)
    scaler_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=1)

    kt_dfb = ttl.make_dataflow_buffer_like(K, shape=(HEAD_TILES, SEQ_TILES), buffer_factor=2)
    qk_dfb = ttl.make_dataflow_buffer_like(Q, shape=(SEQ_TILES, SEQ_TILES), buffer_factor=2)
    scaled_dfb = ttl.make_dataflow_buffer_like(Q, shape=(SEQ_TILES, SEQ_TILES), buffer_factor=2)
    max_dfb = ttl.make_dataflow_buffer_like(Q, shape=(SEQ_TILES, SEQ_TILES), buffer_factor=2)
    max_bcast_dfb = ttl.make_dataflow_buffer_like(Q, shape=(SEQ_TILES, SEQ_TILES), buffer_factor=2)
    exp_dfb = ttl.make_dataflow_buffer_like(Q, shape=(SEQ_TILES, SEQ_TILES), buffer_factor=2)
    sum_dfb = ttl.make_dataflow_buffer_like(Q, shape=(SEQ_TILES, SEQ_TILES), buffer_factor=2)
    sum_bcast_dfb = ttl.make_dataflow_buffer_like(Q, shape=(SEQ_TILES, SEQ_TILES), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(SEQ_TILES, HEAD_TILES), buffer_factor=1)

    @ttl.compute()
    def compute():
        # Step 1: K^T = transpose(K)
        with k_dfb.wait() as kv, kt_dfb.reserve() as kt:
            kt.store(ttl.math.transpose(kv))

        # Step 2: QK = Q @ K^T
        with q_dfb.wait() as qv, kt_dfb.wait() as ktv, qk_dfb.reserve() as qk:
            qk.store(qv @ ktv)

        # Step 3: QK_scaled = QK * scale
        with scale_dfb.wait() as s, qk_dfb.wait() as qkv:
            with scaled_dfb.reserve() as scd:
                bcast = ttl.math.broadcast(s, dims=[0, 1])
                scd.store(bcast * qkv)

        # Steps 4-6: Row-wise softmax
        with scaled_dfb.wait() as sdv, scaler_dfb.wait() as sc:
            with max_dfb.reserve() as mx:
                mx.store(ttl.math.reduce_max(sdv, sc, dims=[1]))
            with max_dfb.wait() as mxv, max_bcast_dfb.reserve() as mxb:
                mxb.store(ttl.math.broadcast(mxv, dims=[1]))
            with max_bcast_dfb.wait() as mxbv:
                with exp_dfb.reserve() as ex:
                    ex.store(ttl.math.exp(sdv - mxbv))
                with exp_dfb.wait() as exv, sum_dfb.reserve() as sm:
                    sm.store(ttl.math.reduce_sum(exv, sc, dims=[1]))
                with sum_dfb.wait() as smv, sum_bcast_dfb.reserve() as smb:
                    smb.store(ttl.math.broadcast(smv, dims=[1]))
                with sum_bcast_dfb.wait() as smbv, qk_dfb.reserve() as attn:
                    attn.store(ttl.math.exp(sdv - mxbv) / smbv)

        # Step 7: out = attn @ V
        with qk_dfb.wait() as av, v_dfb.wait() as vv, out_dfb.reserve() as o:
            o.store(av @ vv)

    @ttl.datamovement()
    def dm_read():
        with q_dfb.reserve() as blk:
            tx = ttl.copy(Q[0:SEQ_TILES, 0:HEAD_TILES], blk); tx.wait()
        with k_dfb.reserve() as blk:
            tx = ttl.copy(K[0:SEQ_TILES, 0:HEAD_TILES], blk); tx.wait()
        with v_dfb.reserve() as blk:
            tx = ttl.copy(V[0:SEQ_TILES, 0:HEAD_TILES], blk); tx.wait()
        with scale_dfb.reserve() as blk:
            tx = ttl.copy(scale[0, 0], blk); tx.wait()
        with scaler_dfb.reserve() as blk:
            tx = ttl.copy(scaler[0, 0], blk); tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[0:SEQ_TILES, 0:HEAD_TILES]); tx.wait()
```

From `sdpa_kernel.py`. Key points:
- Row-wise softmax uses `dims=[1]` to collapse columns (per-row results), then `dims=[1]` broadcast to replicate back
- Transpose, matmul, reduce, broadcast each need their own `with` block and `store`
- `sdv` and `mxbv` are kept in scope via nesting so exp is recomputed rather than stored twice
- Single-core is fine for prototyping the compute pattern; add `grid="auto"` + streaming loops for production

### Example 4: Streaming Gating Kernel (grid="auto", RMSNorm, Reduce/Broadcast)

Real-world kernel from Engram model. Uses `grid="auto"` with `(1,1)` tile DFBs, streaming over sequence tiles. Shows RMSNorm via tile-by-tile reduce accumulation (looping over HIDDEN_TILES), dot product, gating, and the init-then-accumulate pattern for thread-local accumulators.

```python
TILE = 32
HIDDEN_TILES = 32  # 1024-dim / 32

@ttl.kernel(grid="auto")
def engram_gate_kernel(key, query, value, key_norm_w, query_norm_w,
                       scaler, mean_scale, inv_sqrt_d, eps_tile, out):
    grid_cols, _ = ttl.grid_size(dims=2)
    seq_tiles = key.shape[0] // TILE
    tiles_per_core = -(-seq_tiles // grid_cols)

    key_dfb = ttl.make_dataflow_buffer_like(key, shape=(1, 1), buffer_factor=2)
    query_dfb = ttl.make_dataflow_buffer_like(query, shape=(1, 1), buffer_factor=2)
    value_dfb = ttl.make_dataflow_buffer_like(value, shape=(1, 1), buffer_factor=2)
    knw_dfb = ttl.make_dataflow_buffer_like(key_norm_w, shape=(1, 1), buffer_factor=2)
    qnw_dfb = ttl.make_dataflow_buffer_like(query_norm_w, shape=(1, 1), buffer_factor=2)
    scaler_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=1)
    ms_dfb = ttl.make_dataflow_buffer_like(mean_scale, shape=(1, 1), buffer_factor=1)
    isd_dfb = ttl.make_dataflow_buffer_like(inv_sqrt_d, shape=(1, 1), buffer_factor=1)
    eps_dfb = ttl.make_dataflow_buffer_like(eps_tile, shape=(1, 1), buffer_factor=1)

    sq_dfb = ttl.make_dataflow_buffer_like(key, shape=(1, 1), buffer_factor=2)
    red_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    acc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    bcast_dfb = ttl.make_dataflow_buffer_like(key, shape=(1, 1), buffer_factor=2)
    dot_dfb = ttl.make_dataflow_buffer_like(key, shape=(1, 1), buffer_factor=2)
    gate_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        core_x, _ = ttl.node(dims=2)
        with scaler_dfb.wait() as sc, ms_dfb.wait() as ms, isd_dfb.wait() as isd, eps_dfb.wait() as eps:
            for local_t in range(tiles_per_core):
                tile_idx = core_x * tiles_per_core + local_t
                if tile_idx < seq_tiles:
                    # RMSNorm pass 1: sum of squares over HIDDEN_TILES
                    with key_dfb.wait() as k0:
                        with sq_dfb.reserve() as sq:
                            sq.store(k0 * k0)
                    with sq_dfb.wait() as sqv, red_dfb.reserve() as r:
                        r.store(ttl.math.reduce_sum(sqv, sc, dims=[1]))
                    with red_dfb.wait() as rv, acc_dfb.reserve() as acc:
                        acc.store(rv)
                    for j in range(HIDDEN_TILES - 1):
                        with key_dfb.wait() as kj:
                            with sq_dfb.reserve() as sq:
                                sq.store(kj * kj)
                        with sq_dfb.wait() as sqv, red_dfb.reserve() as r:
                            r.store(ttl.math.reduce_sum(sqv, sc, dims=[1]))
                        with red_dfb.wait() as rv, acc_dfb.wait() as av, acc_dfb.reserve() as new_acc:
                            new_acc.store(av + rv)

                    # Broadcast + rsqrt for normalization factor
                    with acc_dfb.wait() as total, bcast_dfb.reserve() as bc:
                        bc.store(ttl.math.broadcast(total, dims=[1]))
                    with bcast_dfb.wait() as bv, red_dfb.reserve() as scaled:
                        scaled.store(bv * ms)
                    with red_dfb.wait() as msq, red_dfb.reserve() as rsq:
                        rsq.store(ttl.math.rsqrt(msq))

                    # ... (query RMSNorm, dot product, gate, gate*value follow same pattern)

    @ttl.datamovement()
    def dm_read():
        core_x, _ = ttl.node(dims=2)
        # Load constants once (buffer_factor=1)
        with scaler_dfb.reserve() as blk:
            tx = ttl.copy(scaler[0, 0], blk); tx.wait()
        with ms_dfb.reserve() as blk:
            tx = ttl.copy(mean_scale[0, 0], blk); tx.wait()
        with isd_dfb.reserve() as blk:
            tx = ttl.copy(inv_sqrt_d[0, 0], blk); tx.wait()
        with eps_dfb.reserve() as blk:
            tx = ttl.copy(eps_tile[0, 0], blk); tx.wait()
        # Stream tiles per core
        for local_t in range(tiles_per_core):
            tile_idx = core_x * tiles_per_core + local_t
            if tile_idx < seq_tiles:
                for j in range(HIDDEN_TILES):
                    with key_dfb.reserve() as blk:
                        tx = ttl.copy(key[tile_idx, j], blk); tx.wait()
                # ... (query, key+weights interleaved, value tiles follow)

    @ttl.datamovement()
    def dm_write():
        core_x, _ = ttl.node(dims=2)
        for local_t in range(tiles_per_core):
            tile_idx = core_x * tiles_per_core + local_t
            if tile_idx < seq_tiles:
                for j in range(HIDDEN_TILES):
                    with out_dfb.wait() as blk:
                        tx = ttl.copy(blk, out[tile_idx, j]); tx.wait()
```

From `engram_demo_ttlang.py`. Key patterns:
- **`grid="auto"` + streaming**: `tiles_per_core = -(-seq_tiles // grid_cols)` handles any sequence length
- **Init-then-accumulate**: First tile initializes `acc_dfb` via `reserve()`, remaining tiles do `wait()` + `reserve()` self-cycle (Rule 3)
- **`(1,1)` tile DFBs with loops**: Iterates over `HIDDEN_TILES` to reduce a full row, one tile at a time
- **Constants loaded once**: `buffer_factor=1` DFBs for scaler/eps/etc. loaded in dm_read, held in scope across all iterations in compute
- **dm_read must produce tiles in exact order compute consumes them**: key tiles for RMSNorm, then key+weights interleaved for dot product, etc.

### Example 5: Pipe Convolution (Forward Chain + Streaming)

Dilated 1D convolution using a forward pipe chain. Each core processes its sequence tiles and pipes boundary data to the next core. Shows pipes combined with streaming loops and the SiLU activation pattern.

```python
N_CONV_CORES = 4
HIDDEN_TILES = 32

@ttl.kernel(grid=(N_CONV_CORES, 1))
def pipe_conv_kernel(s0, s1, s2, s3, w0, w1, w2, w3, out):
    seq_tiles = s0.shape[0] // TILE
    tiles_per_core = -(-seq_tiles // N_CONV_CORES)

    pipes = [ttl.Pipe((x, 0), ((x + 1), 0)) for x in range(N_CONV_CORES - 1)]
    net = ttl.PipeNet(pipes)

    s0_dfb = ttl.make_dataflow_buffer_like(s0, shape=(1, HIDDEN_TILES), buffer_factor=2)
    s1_dfb = ttl.make_dataflow_buffer_like(s1, shape=(1, HIDDEN_TILES), buffer_factor=2)
    s2_dfb = ttl.make_dataflow_buffer_like(s2, shape=(1, HIDDEN_TILES), buffer_factor=2)
    s3_dfb = ttl.make_dataflow_buffer_like(s3, shape=(1, HIDDEN_TILES), buffer_factor=2)
    w0_dfb = ttl.make_dataflow_buffer_like(w0, shape=(1, HIDDEN_TILES), buffer_factor=1)
    w1_dfb = ttl.make_dataflow_buffer_like(w1, shape=(1, HIDDEN_TILES), buffer_factor=1)
    w2_dfb = ttl.make_dataflow_buffer_like(w2, shape=(1, HIDDEN_TILES), buffer_factor=1)
    w3_dfb = ttl.make_dataflow_buffer_like(w3, shape=(1, HIDDEN_TILES), buffer_factor=1)
    acc_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, HIDDEN_TILES), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, HIDDEN_TILES), buffer_factor=2)
    bnd_dfb = ttl.make_dataflow_buffer_like(s0, shape=(1, HIDDEN_TILES), buffer_factor=2)

    @ttl.compute()
    def compute():
        core_x, _ = ttl.node(dims=2)
        # Non-zero cores receive boundary tile from previous core first
        if core_x > 0:
            with bnd_dfb.wait() as bnd, acc_dfb.reserve() as ctx:
                ctx.store(bnd)
            with acc_dfb.wait() as ctx, out_dfb.reserve() as o:
                o.store(ctx)
        # Weighted sum of 4 shifted inputs + SiLU activation
        with w0_dfb.wait() as cw0, w1_dfb.wait() as cw1, w2_dfb.wait() as cw2, w3_dfb.wait() as cw3:
            for local_t in range(tiles_per_core):
                tile_idx = core_x * tiles_per_core + local_t
                if tile_idx < seq_tiles:
                    with s0_dfb.wait() as v0, s1_dfb.wait() as v1, s2_dfb.wait() as v2, s3_dfb.wait() as v3:
                        with acc_dfb.reserve() as acc:
                            acc.store(cw0 * v0 + cw1 * v1 + cw2 * v2 + cw3 * v3)
                    with acc_dfb.wait() as x, out_dfb.reserve() as o:
                        o.store(x * ttl.math.sigmoid(x))  # SiLU

    @ttl.datamovement()
    def dm_read():
        core_x, _ = ttl.node(dims=2)
        if core_x > 0:
            with bnd_dfb.reserve() as blk:
                def recv(pipe):
                    xf = ttl.copy(pipe, blk); xf.wait()
                net.if_dst(recv)
        # Load weights once
        with w0_dfb.reserve() as blk:
            tx = ttl.copy(w0[0, 0:HIDDEN_TILES], blk); tx.wait()
        with w1_dfb.reserve() as blk:
            tx = ttl.copy(w1[0, 0:HIDDEN_TILES], blk); tx.wait()
        with w2_dfb.reserve() as blk:
            tx = ttl.copy(w2[0, 0:HIDDEN_TILES], blk); tx.wait()
        with w3_dfb.reserve() as blk:
            tx = ttl.copy(w3[0, 0:HIDDEN_TILES], blk); tx.wait()
        # Stream sequence tiles, pipe last tile's input to next core
        for local_t in range(tiles_per_core):
            tile_idx = core_x * tiles_per_core + local_t
            if tile_idx < seq_tiles:
                with s0_dfb.reserve() as blk:
                    tx = ttl.copy(s0[tile_idx, 0:HIDDEN_TILES], blk); tx.wait()
                    if local_t == tiles_per_core - 1:
                        if core_x < N_CONV_CORES - 1:
                            def send(pipe):
                                xf = ttl.copy(blk, pipe); xf.wait()
                            net.if_src(send)
                with s1_dfb.reserve() as blk:
                    tx = ttl.copy(s1[tile_idx, 0:HIDDEN_TILES], blk); tx.wait()
                with s2_dfb.reserve() as blk:
                    tx = ttl.copy(s2[tile_idx, 0:HIDDEN_TILES], blk); tx.wait()
                with s3_dfb.reserve() as blk:
                    tx = ttl.copy(s3[tile_idx, 0:HIDDEN_TILES], blk); tx.wait()

    @ttl.datamovement()
    def dm_write():
        core_x, _ = ttl.node(dims=2)
        if core_x > 0:
            prev_tile = core_x * tiles_per_core - 1
            with out_dfb.wait() as blk:
                tx = ttl.copy(blk, out[prev_tile, 0:HIDDEN_TILES]); tx.wait()
        for local_t in range(tiles_per_core):
            tile_idx = core_x * tiles_per_core + local_t
            if tile_idx < seq_tiles:
                with out_dfb.wait() as blk:
                    tx = ttl.copy(blk, out[tile_idx, 0:HIDDEN_TILES]); tx.wait()
```

From `engram_demo_ttlang.py`. Key patterns:
- **Forward pipe chain**: each core pipes its last input tile to the next core for boundary handling
- **Pipe + streaming combined**: boundary receive happens before the main loop, send happens on the last iteration
- **Weights held in scope**: `buffer_factor=1` weights loaded once, kept in scope via outer `with` block across entire streaming loop
- **SiLU activation**: `x * sigmoid(x)` fused in one expression

### Example 6: Full MD Force Kernel (Real-World, 28 DFBs, Factory Pattern)

Complete cell-list molecular dynamics force kernel from a validated simulation (10K atoms, 10K steps, 1.1ms/step). Computes erfc-damped Coulomb + LJ 12-6 forces for all atom pairs across 27 neighbor cells. Shows: factory function pattern for parameterizing kernels with runtime constants, 28 DFBs near the hardware limit, broadcast+transpose for pairwise distance matrices, Horner polynomial evaluation, PBC minimum image convention, and init-then-accumulate force accumulators.

```python
TILE = 32
N_NBR = 27

def make_force_kernel(c_n_dim, c_dim2):
    """Factory: captures cell-grid dimensions as compile-time constants."""

    # Physics constants captured by closure
    c_box = float(box_length)
    c_inv_box = 1.0 / float(box_length)
    c_half = 0.5
    c_lj_scale = 24.0
    c_alpha_sq = float(alpha * alpha)
    c_p_alpha = float(ERFC_P * alpha)
    c_two_a_sp = float(2.0 * alpha / np.sqrt(np.pi))
    c_a1 = float(ERFC_A1)
    c_a2 = float(-ERFC_A2)
    c_a3 = float(ERFC_A3)
    c_a4 = float(-ERFC_A4)
    c_a5 = float(ERFC_A5)

    @ttl.kernel(grid="auto")
    def cell_forces_kernel(own_px, own_py, own_pz, own_q,
                           self_mask, scaler,
                           fx_out, fy_out, fz_out):
        grid_cols, _ = ttl.grid_size(dims=2)
        n_cells = own_px.shape[0] // TILE
        cells_per_core = -(-n_cells // grid_cols)

        ox_cb = ttl.make_dataflow_buffer_like(own_px, shape=(1, 1), buffer_factor=2)
        oy_cb = ttl.make_dataflow_buffer_like(own_py, shape=(1, 1), buffer_factor=2)
        oz_cb = ttl.make_dataflow_buffer_like(own_pz, shape=(1, 1), buffer_factor=2)
        oq_cb = ttl.make_dataflow_buffer_like(own_q, shape=(1, 1), buffer_factor=2)
        ex_cb = ttl.make_dataflow_buffer_like(own_px, shape=(1, 1), buffer_factor=2)
        ey_cb = ttl.make_dataflow_buffer_like(own_py, shape=(1, 1), buffer_factor=2)
        ez_cb = ttl.make_dataflow_buffer_like(own_pz, shape=(1, 1), buffer_factor=2)
        eq_cb = ttl.make_dataflow_buffer_like(own_q, shape=(1, 1), buffer_factor=2)
        sm_cb = ttl.make_dataflow_buffer_like(self_mask, shape=(1, 1), buffer_factor=2)
        sc_cb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)

        ba_cb = ttl.make_dataflow_buffer_like(own_px, shape=(1, 1), buffer_factor=2)
        tr_cb = ttl.make_dataflow_buffer_like(own_px, shape=(1, 1), buffer_factor=2)
        bb_cb = ttl.make_dataflow_buffer_like(own_px, shape=(1, 1), buffer_factor=2)
        r2_tmp = ttl.make_dataflow_buffer_like(own_px, shape=(1, 1), buffer_factor=2)

        r2_cb = ttl.make_dataflow_buffer_like(own_px, shape=(1, 1), buffer_factor=2)
        qq_cb = ttl.make_dataflow_buffer_like(own_px, shape=(1, 1), buffer_factor=2)
        dx_cb = ttl.make_dataflow_buffer_like(own_px, shape=(1, 1), buffer_factor=2)
        dy_cb = ttl.make_dataflow_buffer_like(own_px, shape=(1, 1), buffer_factor=2)
        dz_cb = ttl.make_dataflow_buffer_like(own_px, shape=(1, 1), buffer_factor=2)

        fm_cb = ttl.make_dataflow_buffer_like(own_px, shape=(1, 1), buffer_factor=2)
        ft_cb = ttl.make_dataflow_buffer_like(own_px, shape=(1, 1), buffer_factor=2)
        fr_cb = ttl.make_dataflow_buffer_like(own_px, shape=(1, 1), buffer_factor=2)

        ax_cb = ttl.make_dataflow_buffer_like(fx_out, shape=(1, 1), buffer_factor=2)
        ay_cb = ttl.make_dataflow_buffer_like(fy_out, shape=(1, 1), buffer_factor=2)
        az_cb = ttl.make_dataflow_buffer_like(fz_out, shape=(1, 1), buffer_factor=2)

        fxo_cb = ttl.make_dataflow_buffer_like(fx_out, shape=(1, 1), buffer_factor=2)
        fyo_cb = ttl.make_dataflow_buffer_like(fy_out, shape=(1, 1), buffer_factor=2)
        fzo_cb = ttl.make_dataflow_buffer_like(fz_out, shape=(1, 1), buffer_factor=2)

        @ttl.compute()
        def compute():
            core_x, _ = ttl.node(dims=2)
            for local_c in range(cells_per_core):
                cell_id = core_x * cells_per_core + local_c
                if cell_id < n_cells:
                    with ox_cb.wait() as ox, oy_cb.wait() as oy, oz_cb.wait() as oz, oq_cb.wait() as oq:
                        with sc_cb.wait() as sc:
                            for nbr_i in range(N_NBR):
                                with ex_cb.wait() as ex, ey_cb.wait() as ey, ez_cb.wait() as ez, eq_cb.wait() as eq, sm_cb.wait() as sm:
                                    # PBC pairwise x-distances via broadcast+transpose
                                    with ba_cb.reserve() as ba:
                                        ba.store(ttl.math.broadcast(ox, dims=[1]))
                                    with tr_cb.reserve() as tr:
                                        tr.store(ttl.transpose(ex))
                                    with tr_cb.wait() as trv, bb_cb.reserve() as bb:
                                        bb.store(ttl.math.broadcast(trv, dims=[0]))
                                    with ba_cb.wait() as bav, bb_cb.wait() as bbv:
                                        dx_raw = bav - bbv
                                        dx_pbc = dx_raw - ttl.math.fill(bav, c_box) * ttl.math.floor(dx_raw * ttl.math.fill(bav, c_inv_box) + ttl.math.fill(bav, c_half))
                                        with r2_tmp.reserve() as r2o:
                                            r2o.store(dx_pbc * dx_pbc)
                                        with dx_cb.reserve() as dxo:
                                            dxo.store(dx_pbc)

                                    # PBC pairwise y-distances
                                    with ba_cb.reserve() as ba:
                                        ba.store(ttl.math.broadcast(oy, dims=[1]))
                                    with tr_cb.reserve() as tr:
                                        tr.store(ttl.transpose(ey))
                                    with tr_cb.wait() as trv, bb_cb.reserve() as bb:
                                        bb.store(ttl.math.broadcast(trv, dims=[0]))
                                    with ba_cb.wait() as bav, bb_cb.wait() as bbv, r2_tmp.wait() as r2p:
                                        dy_raw = bav - bbv
                                        dy_pbc = dy_raw - ttl.math.fill(bav, c_box) * ttl.math.floor(dy_raw * ttl.math.fill(bav, c_inv_box) + ttl.math.fill(bav, c_half))
                                        with r2_tmp.reserve() as r2o:
                                            r2o.store(r2p + dy_pbc * dy_pbc)
                                        with dy_cb.reserve() as dyo:
                                            dyo.store(dy_pbc)

                                    # PBC pairwise z-distances (adds self-exclusion mask to r2)
                                    with ba_cb.reserve() as ba:
                                        ba.store(ttl.math.broadcast(oz, dims=[1]))
                                    with tr_cb.reserve() as tr:
                                        tr.store(ttl.transpose(ez))
                                    with tr_cb.wait() as trv, bb_cb.reserve() as bb:
                                        bb.store(ttl.math.broadcast(trv, dims=[0]))
                                    with ba_cb.wait() as bav, bb_cb.wait() as bbv, r2_tmp.wait() as r2p:
                                        dz_raw = bav - bbv
                                        dz_pbc = dz_raw - ttl.math.fill(bav, c_box) * ttl.math.floor(dz_raw * ttl.math.fill(bav, c_inv_box) + ttl.math.fill(bav, c_half))
                                        with r2_cb.reserve() as r2o:
                                            r2o.store(r2p + dz_pbc * dz_pbc + sm)
                                        with dz_cb.reserve() as dzo:
                                            dzo.store(dz_pbc)

                                    # Charge products via broadcast+transpose
                                    with ba_cb.reserve() as ba:
                                        ba.store(ttl.math.broadcast(oq, dims=[1]))
                                    with tr_cb.reserve() as tr:
                                        tr.store(ttl.transpose(eq))
                                    with tr_cb.wait() as trv, bb_cb.reserve() as bb:
                                        bb.store(ttl.math.broadcast(trv, dims=[0]))
                                    with ba_cb.wait() as bav, bb_cb.wait() as bbv, qq_cb.reserve() as qqo:
                                        qqo.store(bav * bbv)

                                    # erfc-damped Coulomb + LJ 12-6 force magnitudes
                                    with r2_cb.wait() as r2, qq_cb.wait() as qq:
                                        r_inv = ttl.math.rsqrt(r2)
                                        r2_inv = ttl.math.recip(r2)
                                        r_val = r2 * r_inv
                                        t = ttl.math.recip(r_inv * r_inv * r2 + ttl.math.fill(r2, c_p_alpha) * r_val)
                                        poly = t * (ttl.math.fill(r2, c_a1) + t * (ttl.math.neg(ttl.math.fill(r2, c_a2)) + t * (ttl.math.fill(r2, c_a3) + t * (ttl.math.neg(ttl.math.fill(r2, c_a4)) + t * ttl.math.fill(r2, c_a5)))))
                                        exp_neg = ttl.math.exp(ttl.math.neg(ttl.math.fill(r2, c_alpha_sq) * r2))
                                        erfc_val = poly * exp_neg
                                        with ft_cb.reserve() as coul:
                                            coul.store(qq * (erfc_val * r2_inv + ttl.math.fill(r2, c_two_a_sp) * exp_neg * r_inv) * r_inv)
                                        r2_inv2 = ttl.math.recip(r2)
                                        r4_inv = r2_inv2 * r2_inv2
                                        r6_inv = r4_inv * r2_inv2
                                        r12_inv = r6_inv * r6_inv
                                        with fr_cb.reserve() as lj:
                                            lj.store(ttl.math.fill(r2, c_lj_scale) * r2_inv2 * (r12_inv + r12_inv - r6_inv))

                                    with ft_cb.wait() as fc, fr_cb.wait() as fl:
                                        with fm_cb.reserve() as fmo:
                                            fmo.store(fl + fc)

                                    # Project onto displacements, reduce rows, accumulate per-axis
                                    with fm_cb.wait() as fm:
                                        with dx_cb.wait() as dxv:
                                            with ft_cb.reserve() as ft:
                                                ft.store(fm * dxv)
                                            with ft_cb.wait() as ftv, fr_cb.reserve() as fr:
                                                fr.store(ttl.math.reduce_sum(ftv, sc, dims=[1]))
                                            if nbr_i == 0:
                                                with fr_cb.wait() as frv, ax_cb.reserve() as ax:
                                                    ax.store(frv)
                                            else:
                                                with fr_cb.wait() as frv, ax_cb.wait() as prev:
                                                    with ax_cb.reserve() as ax:
                                                        ax.store(prev + frv)
                                        with dy_cb.wait() as dyv:
                                            with ft_cb.reserve() as ft:
                                                ft.store(fm * dyv)
                                            with ft_cb.wait() as ftv, fr_cb.reserve() as fr:
                                                fr.store(ttl.math.reduce_sum(ftv, sc, dims=[1]))
                                            if nbr_i == 0:
                                                with fr_cb.wait() as frv, ay_cb.reserve() as ay:
                                                    ay.store(frv)
                                            else:
                                                with fr_cb.wait() as frv, ay_cb.wait() as prev:
                                                    with ay_cb.reserve() as ay:
                                                        ay.store(prev + frv)
                                        with dz_cb.wait() as dzv:
                                            with ft_cb.reserve() as ft:
                                                ft.store(fm * dzv)
                                            with ft_cb.wait() as ftv, fr_cb.reserve() as fr:
                                                fr.store(ttl.math.reduce_sum(ftv, sc, dims=[1]))
                                            if nbr_i == 0:
                                                with fr_cb.wait() as frv, az_cb.reserve() as az:
                                                    az.store(frv)
                                            else:
                                                with fr_cb.wait() as frv, az_cb.wait() as prev:
                                                    with az_cb.reserve() as az:
                                                        az.store(prev + frv)

                            with ax_cb.wait() as fx, fxo_cb.reserve() as fxo:
                                fxo.store(fx)
                            with ay_cb.wait() as fy, fyo_cb.reserve() as fyo:
                                fyo.store(fy)
                            with az_cb.wait() as fz, fzo_cb.reserve() as fzo:
                                fzo.store(fz)

        @ttl.datamovement()
        def dm_read():
            core_x, _ = ttl.node(dims=2)
            for local_c in range(cells_per_core):
                cell_id = core_x * cells_per_core + local_c
                if cell_id < n_cells:
                    with ox_cb.reserve() as blk:
                        tx = ttl.copy(own_px[cell_id, 0], blk); tx.wait()
                    with oy_cb.reserve() as blk:
                        tx = ttl.copy(own_py[cell_id, 0], blk); tx.wait()
                    with oz_cb.reserve() as blk:
                        tx = ttl.copy(own_pz[cell_id, 0], blk); tx.wait()
                    with oq_cb.reserve() as blk:
                        tx = ttl.copy(own_q[cell_id, 0], blk); tx.wait()
                    with sc_cb.reserve() as blk:
                        tx = ttl.copy(scaler[0, 0], blk); tx.wait()
                    # Compute neighbor cell IDs from 3D grid coordinates
                    cx = cell_id // c_dim2
                    cy = (cell_id // c_n_dim) % c_n_dim
                    cz = cell_id % c_n_dim
                    for nbr in range(N_NBR):
                        off_dx = (nbr // 9) - 1
                        off_dy = ((nbr // 3) % 3) - 1
                        off_dz = (nbr % 3) - 1
                        nbr_cell = ((cx + off_dx + c_n_dim) % c_n_dim) * c_dim2 + ((cy + off_dy + c_n_dim) % c_n_dim) * c_n_dim + ((cz + off_dz + c_n_dim) % c_n_dim)
                        with ex_cb.reserve() as blk:
                            tx = ttl.copy(own_px[nbr_cell, 0], blk); tx.wait()
                        with ey_cb.reserve() as blk:
                            tx = ttl.copy(own_py[nbr_cell, 0], blk); tx.wait()
                        with ez_cb.reserve() as blk:
                            tx = ttl.copy(own_pz[nbr_cell, 0], blk); tx.wait()
                        with eq_cb.reserve() as blk:
                            tx = ttl.copy(own_q[nbr_cell, 0], blk); tx.wait()
                        with sm_cb.reserve() as blk:
                            tx = ttl.copy(self_mask[cell_id * N_NBR + nbr, 0], blk); tx.wait()

        @ttl.datamovement()
        def dm_write():
            core_x, _ = ttl.node(dims=2)
            for local_c in range(cells_per_core):
                cell_id = core_x * cells_per_core + local_c
                if cell_id < n_cells:
                    with fxo_cb.wait() as blk:
                        tx = ttl.copy(blk, fx_out[cell_id, 0]); tx.wait()
                    with fyo_cb.wait() as blk:
                        tx = ttl.copy(blk, fy_out[cell_id, 0]); tx.wait()
                    with fzo_cb.wait() as blk:
                        tx = ttl.copy(blk, fz_out[cell_id, 0]); tx.wait()

    return cell_forces_kernel

# Usage: kernel is built with runtime cell-grid parameters, then called repeatedly
cell_forces_kernel = make_force_kernel(c_n_dim=n_cells_dim, c_dim2=n_cells_dim**2)
cell_forces_kernel(tt_px, tt_py, tt_pz, tt_q, tt_masks, tt_scaler, tt_fx, tt_fy, tt_fz)
```

From `md_cell_list.py` (validated: 10K atoms, 10K steps, 1.1ms/step). Key patterns:
- **Factory function**: `make_force_kernel(c_n_dim, c_dim2)` captures cell-grid dimensions and physics constants as closure variables. The returned kernel is called many times per MD step with different positions.
- **28 DFBs** near the 32-DFB hardware limit: own cell (4), neighbor cell (5), geometry intermediates (9), force intermediates (3), xyz accumulators (3), output (3), scaler (1).
- **Pairwise distance via broadcast+transpose**: `broadcast(ox, dims=[1])` expands column vector to NxN, `transpose(ex)` + `broadcast(dims=[0])` expands row vector. Subtraction gives all-pairs distance matrix in one tile.
- **PBC minimum image**: `dx - box * floor(dx/box + 0.5)` with `ttl.math.fill` for scalar constants.
- **Self-exclusion mask**: Added directly to r2 so self-pairs and empty slots get r2~1e6, making forces vanish naturally.
- **Horner polynomial for erfc**: Nested multiply-add chain, 20+ fused elementwise ops in a single `with` block.
- **Init-then-accumulate** (CB threading Rule 3): `nbr_i == 0` initializes ax/ay/az via `reserve()`, subsequent neighbors do `wait()` + `reserve()` self-cycle.
- **Computed neighbor indices in dm_read**: 3D grid coordinates derived from flat cell_id, neighbor offsets applied with PBC wrapping. No pre-built index table needed.
- **Deeply nested scoping**: Own cell data held in scope across all 27 neighbor iterations (single `wait()`, reused 27 times).
