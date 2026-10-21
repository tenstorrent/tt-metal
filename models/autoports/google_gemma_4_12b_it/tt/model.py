# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Full ``google/gemma-4-12B-it`` text model on a QB2 (four Blackhole chips, ``[1, 4]`` ring mesh, tensor parallel 4).

The block stack is the optimized multichip decoder (``multichip_decoder.MultichipDecoder``): its per-layer precision
policy (BFP8 / per-layer BFP4 weights, LoFi decode matmuls, BFP8 paged KV cache, BFP8 decode all-reduce payloads, BFP8
gate/up prefill gather), its CCL plan and its inter-layer residual layouts are used unchanged:

* Decode: ``[1, 1, B, 3840]`` BF16 **replicated** on every chip in the layers' L1 residual layout (15 cores). The
  embedding output is gathered once at the stack entry; the final norm and LM head read the replicated residual.
* Prefill: ``[1, 1, S, 960]`` BF16 per chip, **hidden-sharded** (chip d holds hidden columns ``[960 d, 960 (d + 1))``).
  The embedding table is stored hidden-sharded, so its lookup *is* the prefill residual layout; only the selected
  last-token rows are gathered for the final norm and LM head.

Terminal path (``decode_logits``): final RMSNorm on the replicated residual (sharded LayerNorm program of the layers),
LM head against the tied embedding in the precision config's dtype (BFP8: 1D-multicast matmul over the full grid on the
DRAM-interleaved weight; BFP4: DRAM-sharded matmul, 3 readers per bank, ``LM_HEAD_POLICY``),
vocab-sharded across the chips (chip d owns tokens
``[65536 d, 65536 (d + 1))``), and the final logit soft-cap ``30 * tanh(x / 30)`` fused into one unary chain that writes
the caller's persistent ``[1, 1, 32, 65536]`` per-chip logits buffer. Logits are never gathered across chips; the
sampler (``tt/generator.py``) reduces each chip's slice to 32 candidates first.

Prefill runs chunk-major: every chunk of at most ``prefill_chunk`` (8192) prompt tokens goes through all 48 layers
before the next chunk, so each layer call sees one chunk at a chunk-aligned start. The last chunk is zero-padded to a
power-of-two bucket (128..8192 rows), which keeps the set of compiled prefill programs finite:
``{chunk starts 0, 8192, ...} x {bucket lengths}``. ``warm_prefill`` compiles all of them before any trace capture.
Padded rows are causally invisible to real rows; their K/V land in the request's own last block or in the
scratch block (``prefill_page_table`` maps the padded tail there) and are overwritten by decode before they could
be attended.

KV cache ownership. Standalone (``num_blocks`` > 0, the default): every layer allocates its paged ``[k, v]`` at
construction; block 0 is the zero canary, block 1 the scratch block, requests use ``[2, num_blocks)``. Serving
(``num_blocks=0``): the layers are built without caches and the owner (vLLM, ``tt/generator_vllm.py``) attaches
them with ``attach_kv_cache``: block 0 is vLLM's null block (the canary), requests use ``[1, n)`` and the extra
physical block ``n`` is the scratch block. Page tables may differ per layer (vLLM hybrid KV groups: sliding-window
layers hold only their window's blocks, the rest map to block 0); every page-table argument below is either one
table for all layers or a list with one table per layer of ``self.layers``.
"""

import os
from pathlib import Path

import torch

import ttnn
from models.autoports.google_gemma_4_12b_it.tt.functional_decoder import MAX_DECODE_BATCH, TILE
from models.autoports.google_gemma_4_12b_it.tt.fused_decoder import DEFAULT_PREFILL_CHUNK
from models.autoports.google_gemma_4_12b_it.tt.multichip_decoder import MESH_SHAPE, TP, MultichipDecoder, mesh_ccl
from models.autoports.google_gemma_4_12b_it.tt.optimized_decoder import FIDELITY
from models.autoports.google_gemma_4_12b_it.tt.precision import PrecisionConfig, built_precision_summary, deep_merge

HF_MODEL = "google/gemma-4-12B-it"
HF_REVISION = os.environ.get("HF_REVISION", "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7")
TEXT_PREFIX = "model.language_model."

BLOCK_SIZE = 64
CANARY_BLOCK = 0  # zero block: every page-table entry outside a request names it; never written
SCRATCH_BLOCK = 1  # bucket-padding rows and warmup write here; never attended by a real query
RESERVED_BLOCKS = 2
PREFILL_BUCKETS = (128, 256, 512, 1024, 2048, 4096, 8192)

# LM head (tied embedding, vocab-sharded, [3840, 65536] per chip). dtype / fidelity come from the precision config
# (tt/precision.py); the program follows the dtype unless ``lm_head`` overrides "program":
# * BFP8: a 1D-multicast program over the 11 x 10 grid (19 output tiles per core) reading the DRAM-interleaved weight:
#   720 us vs 878 / 761 us for the DRAM-sharded program with 2 / 3 readers per bank (doc/optimized_full_model/logs/
#   lm_head_probe3.json);
# * BFP4: the DRAM-sharded program with 3 reader cores per bank (columns padded to 66048 per chip so each bank shard
#   splits over its readers) and in0_block_w 1 on the L1 width-sharded normalized hidden state: 403 us vs 525 us for the
#   best 1D-multicast geometry; 4 readers exceed the op's limit (workers_per_bank <= 3), in0_block_w 8 overflows L1
#   (doc/datatype_sweep/logs/lm_head_bfp4_probe.json).
LM_HEAD_POLICY = {"dtype": ttnn.bfloat8_b, "fidelity": "hifi2", "in0_block_w": 8, "program": None, "readers": 3}
LM_HEAD_PROGRAM_BY_DTYPE = {ttnn.bfloat8_b: "mcast1d", ttnn.bfloat4_b: "dram_sharded", ttnn.bfloat16: "mcast1d"}
_DTYPES = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}


def checkpoint_dir():
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            HF_MODEL, revision=HF_REVISION, allow_patterns=["*.json", "*.safetensors"], local_files_only=True
        )
    )


def text_config():
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(HF_MODEL, revision=HF_REVISION).text_config
    cfg._attn_implementation = "eager"
    return cfg


class Checkpoint:
    """Lazy reader of the single-file BF16 checkpoint (text decoder tensors only)."""

    def __init__(self, path=None):
        from safetensors import safe_open

        path = Path(path) if path is not None else checkpoint_dir()
        self.file = safe_open(str(path / "model.safetensors"), "pt")

    def tensor(self, name):
        return self.file.get_tensor(TEXT_PREFIX + name)

    def layer(self, layer_idx):
        prefix = f"{TEXT_PREFIX}layers.{layer_idx}."
        out = {k[len(prefix) :]: self.file.get_tensor(k) for k in self.file.keys() if k.startswith(prefix)}
        assert out, f"no checkpoint tensors for layer {layer_idx}"
        return out


def prefill_bucket(rows, chunk=DEFAULT_PREFILL_CHUNK):
    """Physical row count of a prefill chunk with ``rows`` real tokens (power-of-two bucket, 128..chunk)."""
    assert 1 <= rows <= chunk
    return next(b for b in PREFILL_BUCKETS if b >= rows and b <= chunk) if rows <= PREFILL_BUCKETS[-1] else chunk


class Gemma4Model:
    """Embedding, the 48 multichip decoder layers, final norm and LM head of Gemma 4 12B on the QB2 ring.

    ``layer_indices`` builds a reduced model (e.g. ``[0, 5]``: one sliding and one full-attention layer with their
    real weights, shapes and policies) for profiling and fast debugging; the terminal path is always complete.
    """

    def __init__(
        self,
        mesh_device,
        *,
        num_blocks=None,
        max_context=None,
        layer_indices=None,
        policy=None,
        lm_head=None,
        precision=None,
        prefill_chunk=DEFAULT_PREFILL_CHUNK,
        persistent_entry_gather=False,
        checkpoint=None,
        hf_config=None,
        log=print,
    ):
        assert tuple(mesh_device.shape) == MESH_SHAPE, f"Gemma 4 12B runs on a {MESH_SHAPE} QB2 mesh"
        self.mesh = mesh_device
        self.config = c = hf_config or text_config()
        assert c.tie_word_embeddings and c.hidden_size_per_layer_input == 0
        self.vocab_size = c.vocab_size
        self.hidden_size = c.hidden_size
        assert self.vocab_size % (TP * TILE) == 0
        self.vocab_per_chip = self.vocab_size // TP
        self.max_context = c.max_position_embeddings if max_context is None else max_context
        assert self.max_context % prefill_chunk == 0 and prefill_chunk % BLOCK_SIZE == 0
        self.block_size = BLOCK_SIZE
        self.pages_per_user = self.max_context // BLOCK_SIZE
        self.num_blocks = self.pages_per_user + RESERVED_BLOCKS if num_blocks is None else num_blocks
        assert self.num_blocks == 0 or self.num_blocks > RESERVED_BLOCKS
        # block bookkeeping (``attach_kv_cache`` rebinds it): request blocks are [first_block, block_limit)
        self.canary_block = CANARY_BLOCK
        self.scratch_block = SCRATCH_BLOCK
        self.first_block = RESERVED_BLOCKS
        self.block_limit = self.num_blocks
        self.prefill_chunk = prefill_chunk
        self.softcap = c.final_logit_softcapping
        self.eps = c.rms_norm_eps
        self.layer_indices = list(range(c.num_hidden_layers)) if layer_indices is None else list(layer_indices)
        ckpt = checkpoint if isinstance(checkpoint, Checkpoint) else Checkpoint(checkpoint)
        # precision policy (tt/precision.py): the selected datatype-sweep config unless ``precision`` names another;
        # ``policy`` / ``lm_head`` stay as explicit experiment overrides on top of it
        self.precision = PrecisionConfig.load(precision, c.num_hidden_layers)
        log(f"precision config: {self.precision.config_id} ({self.precision.source})")
        replicate = ttnn.ReplicateTensorToMesh(mesh_device)

        # ---- decoder stack (the layer owns its paged K/V cache: [num_blocks, local KV heads, 64, head_dim] BFP8)
        shared = {}
        self.layers = []
        for i in self.layer_indices:
            sd = ckpt.layer(i)
            self.layers.append(
                MultichipDecoder.from_state_dict(
                    sd,
                    hf_config=c,
                    layer_idx=i,
                    mesh_device=mesh_device,
                    num_blocks=self.num_blocks,
                    block_size=BLOCK_SIZE,
                    prefill_chunk_size=prefill_chunk,
                    max_context=self.max_context,
                    shared=shared,
                    policy=deep_merge(self.precision.layer_policy(i, c.layer_types[i]), policy),
                )
            )
            del sd
            log(f"layer {i:2d} ({self.layers[-1].layer_type}) built")
        self.ccl = mesh_ccl(mesh_device)
        first = self.layers[0]
        self.grid = first.grid
        self.residual_mem = first.residual_mem
        self.residual_norm_cfg = first.residual_norm_cfg
        self.compute_hifi4_fp32 = first.compute_hifi4_fp32
        # one layer per kind computes the per-step shared decode inputs (RoPE rows, user-split slices)
        self.kind_layer = {}
        for layer in self.layers:
            self.kind_layer.setdefault(layer.layer_type, layer)
        self.full_layer = self.kind_layer.get("full_attention")
        self.full_layer_pos = self.layers.index(self.full_layer) if self.full_layer is not None else None

        # ---- embedding: tied table scaled by bf16(sqrt(hidden)) (HF: bf16 lookup * embed_scale.to(bf16), a single
        # rounding of the exact product, so folding the scale into the table is bit-identical), hidden-sharded rows.
        emb = ckpt.tensor("embed_tokens.weight")
        assert tuple(emb.shape) == (self.vocab_size, self.hidden_size) and emb.dtype == torch.bfloat16
        scale = torch.tensor(self.hidden_size**0.5, dtype=torch.bfloat16).float()
        self.embed_scale = float(scale)
        self.embedding = ttnn.from_torch(
            (emb.float() * scale).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=1),
        )

        # ---- LM head: emb.T [3840, 262144], vocab-sharded (chip d: [3840, 65536])
        lp = dict(LM_HEAD_POLICY, **self.precision.lm_head_policy())
        lp.update(lm_head or {})
        lp["dtype"] = _DTYPES.get(lp["dtype"], lp["dtype"])
        lp["program"] = lp["program"] or LM_HEAD_PROGRAM_BY_DTYPE[lp["dtype"]]
        self.lm_policy = lp
        n = self.vocab_per_chip
        if lp["program"] == "dram_sharded":
            # weight width-sharded over the DRAM banks, each chip's columns padded so a bank shard splits evenly
            # over its reader cores (the padded logits are zero and are sliced off before the soft-cap)
            banks = mesh_device.dram_grid_size().x
            readers = lp["readers"]
            lp["in0_block_w"] = 1  # the only width that fits L1 (see LM_HEAD_POLICY); recorded so reports show it
            tiles_per_reader = -(-(n // TILE) // (banks * readers))
            self.lm_cols = banks * readers * tiles_per_reader * TILE
            w = emb.T.contiguous()
            if self.lm_cols != n:
                w = torch.cat(
                    [
                        torch.cat(
                            [w[:, d * n : (d + 1) * n], torch.zeros(self.hidden_size, self.lm_cols - n, dtype=w.dtype)],
                            1,
                        )
                        for d in range(TP)
                    ],
                    1,
                )
            self.lm_head = ttnn.from_torch(
                w,
                dtype=lp["dtype"],
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                    ttnn.BufferType.DRAM,
                    ttnn.ShardSpec(
                        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))}),
                        (self.hidden_size, self.lm_cols // banks),
                        ttnn.ShardOrientation.ROW_MAJOR,
                    ),
                ),
                mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=1),
            )
            del w
            workers = banks * readers
            self.lm_cfg = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                in0_block_w=lp["in0_block_w"],
                per_core_M=1,
                per_core_N=tiles_per_reader,
                fused_activation=None,
                num_workers_per_dram_bank=readers,
            )
            self.lm_out_mem = ttnn.create_sharded_memory_config(
                (TILE, self.lm_cols // workers),
                core_grid=ttnn.num_cores_to_corerangeset(workers, self.grid, row_wise=True),
                strategy=ttnn.ShardStrategy.WIDTH,
                use_height_and_width_as_shard_shape=True,
            )
        else:
            self.lm_cols = n
            self.lm_head = ttnn.from_torch(
                emb.T.contiguous(),
                dtype=lp["dtype"],
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=1),
            )
            cores = self.grid.x * self.grid.y
            per_core_n = -(-(n // TILE) // cores)
            self.lm_cfg = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=self.grid,
                in0_block_w=lp["in0_block_w"],
                out_subblock_h=1,
                out_subblock_w=1,
                out_block_h=1,
                out_block_w=per_core_n,
                per_core_M=1,
                per_core_N=per_core_n,
                fuse_batch=True,
                fused_activation=None,
                mcast_in0=True,
            )
        del emb
        fid, fp32, l1acc = FIDELITY[lp["fidelity"]]
        self.lm_ck = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(), math_fidelity=fid, math_approx_mode=False, fp32_dest_acc_en=fp32, packer_l1_acc=l1acc
        )
        self.softcap_chain = [
            ttnn.UnaryWithParam(ttnn.UnaryOpType.MUL_UNARY_SFPU, 1.0 / self.softcap),
            ttnn.UnaryWithParam(ttnn.UnaryOpType.TANH),
            ttnn.UnaryWithParam(ttnn.UnaryOpType.MUL_UNARY_SFPU, self.softcap),
        ]
        # decode entry gather into persistent DRAM outputs per decode batch (as the layers' user-split gathers, with
        # the barrier semaphore): measured equal to the fresh L1 output (1319.4 vs 1320.7 us per reduced-model step,
        # doc/optimized_full_model/logs/ccl_buffers_ab.json), so the default keeps the fresh L1 output
        self.entry_gather_bufs = {}
        if persistent_entry_gather:
            for b in range(1, MAX_DECODE_BATCH + 1):
                self.entry_gather_bufs[b] = ttnn.from_torch(
                    torch.zeros(1, 1, b, self.hidden_size),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh_device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=replicate,
                )
        self.final_norm = ttnn.from_torch(
            ckpt.tensor("norm.weight").reshape(1, 1, -1, TILE).contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=replicate,
        )
        # what was actually built (tensor dtypes, reader counts, compute-kernel fidelities), checked against the
        # precision config unless explicit policy / lm_head overrides were passed for an experiment
        self.precision_summary = built_precision_summary(self)
        self.precision_overridden = bool(policy or lm_head)
        if not self.precision_overridden:
            self.precision.verify(self.precision_summary)

    # ------------------------------------------------------------------ host <-> device helpers
    def kv_cache(self):
        """The paged K/V caches, ``[[k, v] per layer]`` (allocated at construction or attached by the owner)."""
        return [layer.kv_cache for layer in self.layers]

    def attach_kv_cache(self, caches, request_blocks):
        """Bind externally owned caches (vLLM): ``caches[i] = [k, v]`` for ``self.layers[i]``, each
        ``[request_blocks + 1, *layer.kv_block_shape]``. Block 0 is the owner's never-written null block (the canary),
        requests use blocks ``[1, request_blocks)`` and the extra block ``request_blocks`` is the scratch block."""
        assert len(caches) == len(self.layers)
        for layer, (k, v) in zip(self.layers, caches):
            want = [request_blocks + 1, *layer.kv_block_shape]
            assert list(k.shape) == want and list(v.shape) == want, (list(k.shape), want)
            assert k.dtype == layer.kv_dtype and v.dtype == layer.kv_dtype
            layer.kv_cache = [k, v]
            layer.num_blocks = request_blocks + 1
        self.num_blocks = request_blocks + 1
        if not self.precision_overridden:
            self.precision_summary = built_precision_summary(self)
            self.precision.verify(self.precision_summary)
        self.canary_block = 0
        self.first_block = 1
        self.block_limit = request_blocks
        self.scratch_block = request_blocks

    def layer_tables(self, page_table):
        """One page table per layer of ``self.layers`` from a single table or a per-layer list."""
        if isinstance(page_table, (list, tuple)):
            assert len(page_table) == len(self.layers), "one page table per layer"
            return list(page_table)
        return [page_table] * len(self.layers)

    def device_ints(self, t, dtype=ttnn.int32):
        return ttnn.from_torch(
            t.contiguous(),
            dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )

    def logits_buffer(self):
        """Persistent ``[1, 1, 32, 65536]`` per-chip BF16 logits (vocab-sharded), the sampler's input."""
        return ttnn.from_torch(
            torch.zeros(1, 1, MAX_DECODE_BATCH, self.vocab_size),
            dtype=self.precision.logits_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh, dim=3),
        )

    def logits_to_torch(self, logits, rows=None):
        """Host ``[rows, vocab]`` float32 view of a vocab-sharded logits tensor (the host-sampling / logits API)."""
        out = ttnn.to_torch(logits, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=3)).float()
        out = out.reshape(-1, self.vocab_size)
        return out if rows is None else out[:rows]

    def use_user_split(self, batch, max_position):
        """Whether the decode trace for ``batch`` users up to ``max_position`` takes the user-split full layers."""
        return self.full_layer is not None and self.full_layer.use_user_split(batch, max_position)

    # ------------------------------------------------------------------ terminal path
    def decode_logits(self, x, logits_out):
        """Replicated residual ``[1, 1, B, 3840]`` -> soft-capped logits written into ``logits_out`` (rows >= B are
        don't-care padding rows)."""
        h = ttnn.rms_norm(
            x,
            epsilon=self.eps,
            weight=self.final_norm,
            program_config=self.residual_norm_cfg,
            memory_config=self.residual_mem,
            compute_kernel_config=self.compute_hifi4_fp32,
        )
        if self.lm_policy["program"] == "dram_sharded":
            # the DRAM-sharded program reads the L1 width-sharded hidden state as is; its L1 output holds the padded
            # columns, which the slice into DRAM drops
            out = ttnn.linear(
                h,
                self.lm_head,
                program_config=self.lm_cfg,
                compute_kernel_config=self.lm_ck,
                memory_config=self.lm_out_mem,
                dtype=self.precision.logits_dtype,
            )
            ttnn.deallocate(h)
            logits = ttnn.slice(
                out, [0, 0, 0, 0], [1, 1, out.shape[-2], self.vocab_per_chip], memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            ttnn.deallocate(out)
        else:
            h_i = ttnn.to_memory_config(h, ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(h)
            logits = ttnn.linear(
                h_i,
                self.lm_head,
                program_config=self.lm_cfg,
                compute_kernel_config=self.lm_ck,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=self.precision.logits_dtype,
            )
            ttnn.deallocate(h_i)
        rows = logits.shape[-2]
        if rows != MAX_DECODE_BATCH:
            # the matmul output is tile-padded to 32 rows; view it as the buffer's 32 rows (metadata only)
            shape = [1, 1, MAX_DECODE_BATCH, self.vocab_per_chip]
            logits = ttnn.reshape(logits, shape, shape)
        ttnn.unary_chain(logits, self.softcap_chain, output_tensor=logits_out)
        ttnn.deallocate(logits)
        return logits_out

    # ------------------------------------------------------------------ decode
    def embed_decode(self, tokens, batch):
        """Persistent ``[1, 1, 1, 32]`` uint32 token buffer -> replicated ``[1, 1, B, 3840]`` decode residual."""
        idx = ttnn.reshape(tokens, [1, MAX_DECODE_BATCH])
        x = ttnn.embedding(idx, self.embedding, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)  # [1, 32, 960]
        local = self.hidden_size // TP
        x = ttnn.reshape(x, [1, 1, batch, local], [1, 1, MAX_DECODE_BATCH, local])
        buf = self.entry_gather_bufs.get(batch)
        if buf is None:
            g = self.ccl.all_gather(x, dim=3, memory_config=ttnn.L1_MEMORY_CONFIG)
        else:
            g = self.ccl.all_gather(x, dim=3, memory_config=ttnn.DRAM_MEMORY_CONFIG, persistent_output_buffer=buf)
        ttnn.deallocate(x)
        out = ttnn.to_memory_config(g, self.residual_mem)
        if buf is None:
            ttnn.deallocate(g)
        return out

    def decode_step(self, tokens, positions, page_table, logits_out, *, user_split=False, advance=True):
        """One device-only decode step (the body of a decode trace).

        tokens: ``[1, 1, 1, 32]`` uint32 (row b = user b's input token); positions: ``[B]`` int32 (-1 = inactive);
        page_table: ``[B, pages]`` int32, one for all layers or one per layer. Writes soft-capped logits into
        ``logits_out`` and, with ``advance``, increments every active position on device (``plus_one``), so a replay
        loop needs no host position refresh.
        """
        batch = positions.shape[0]
        tables = self.layer_tables(page_table)
        x = self.embed_decode(tokens, batch)
        rows = {kind: layer.decode_rope_rows(positions, batch) for kind, layer in self.kind_layer.items()}
        split = (
            self.full_layer.decode_split_inputs(positions, tables[self.full_layer_pos], user_split=user_split)
            if self.full_layer is not None
            else None
        )
        for layer, table in zip(self.layers, tables):
            y = layer.decode_forward(
                x,
                current_pos=positions,
                page_table=table,
                rope_rows=rows[layer.layer_type],
                split_inputs=split if not layer.is_sliding else None,
                user_split=user_split if not layer.is_sliding else None,
            )
            ttnn.deallocate(x)
            x = y
        for cos, sin in rows.values():
            ttnn.deallocate(cos)
            ttnn.deallocate(sin)
        if split is not None:
            for t in split:
                ttnn.deallocate(t)
        self.decode_logits(x, logits_out)
        ttnn.deallocate(x)
        if advance:
            ttnn.plus_one(positions, skip_negative_entries=True)
        return logits_out

    # ------------------------------------------------------------------ prefill
    def prefill_page_table(self, row, real_end, padded_end, valid_from=0):
        """``[1, pages]`` int32 page-table row for one prefill call: the request's blocks up to ``real_end``, the
        scratch block for blocks that only hold bucket-padding rows (up to ``padded_end``), the canary beyond.
        Blocks ``[valid_from, real_end)`` (by position) must be request blocks; a sliding-window layer's row may hold
        the canary below its window (vLLM frees those blocks), which its attention never reads unmasked."""
        row = torch.as_tensor(row, dtype=torch.int32).reshape(-1)
        assert row.numel() == self.pages_per_user, f"page-table rows need {self.pages_per_user} entries"
        out = torch.full((1, self.pages_per_user), self.canary_block, dtype=torch.int32)
        real_blocks = -(-real_end // BLOCK_SIZE)
        out[0, :real_blocks] = row[:real_blocks]
        out[0, real_blocks : -(-padded_end // BLOCK_SIZE)] = self.scratch_block
        used = out[0, valid_from // BLOCK_SIZE : real_blocks]
        assert ((used >= self.first_block) & (used < self.block_limit)).all(), (
            f"the page-table row must map every block of positions [{valid_from}, {real_end}) to a request block in "
            f"[{self.first_block}, {self.block_limit})"
        )
        return out

    def prefill_chunk_hidden(self, token_ids, page_table, chunk_start):
        """Run one chunk through the stack: ``token_ids`` ``[L]`` (bucket-padded; a host tensor, or a ``[1, L]``
        uint32 device tensor such as a prefill trace's persistent token buffer) at absolute ``chunk_start``.

        Returns this chip's ``[1, 1, L, 960]`` hidden-sharded output of the last layer (before the final norm).
        """
        tables = self.layer_tables(page_table)
        on_host = isinstance(token_ids, torch.Tensor)
        rows = token_ids.numel() if on_host else token_ids.shape[-1]
        assert rows in PREFILL_BUCKETS or rows == self.prefill_chunk, rows
        assert chunk_start % self.prefill_chunk == 0
        tok = self.device_ints(token_ids.reshape(1, rows), dtype=ttnn.uint32) if on_host else token_ids
        x = ttnn.embedding(tok, self.embedding, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)  # [1, L, 960]
        if on_host:
            ttnn.deallocate(tok)
        x = ttnn.reshape(x, [1, 1, rows, self.hidden_size // TP])
        for layer, table in zip(self.layers, tables):
            y = layer.prefill_forward(x, page_table=table, start_pos=chunk_start, user_id=0)
            ttnn.deallocate(x)
            x = y
        return x

    def prefill_request_logits(self, tokens, page_table, row_ids, logits_out):
        """A whole single-chunk request at position 0 (the body of a prefill trace, one per bucket): persistent
        ``[1, L]`` token buffer -> 48 layers -> the rows ``row_ids`` (the last prompt token) -> terminal into
        ``logits_out``. Every input is a persistent device tensor the generator rewrites before a replay."""
        h = self.prefill_chunk_hidden(tokens, page_table, 0)
        x = self.select_rows(h, row_ids)
        ttnn.deallocate(h)
        self.decode_logits(x, logits_out)
        ttnn.deallocate(x)
        return logits_out

    def select_rows(self, hidden, row_ids):
        """Rows ``row_ids`` (``[1, 32]`` uint32 device tensor of row indices) of a ``[1, 1, L, 960]`` hidden-sharded
        prefill output, gathered to the replicated ``[1, 1, 32, 3840]`` decode residual layout. The row indices are
        runtime data, so one program serves every prompt length of a bucket."""
        rows = hidden.shape[-2]
        h_rm = ttnn.to_layout(hidden, ttnn.ROW_MAJOR_LAYOUT)
        h_rm = ttnn.reshape(h_rm, [rows, self.hidden_size // TP])
        sel = ttnn.embedding(row_ids, h_rm, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)  # [1, 32, 960]
        ttnn.deallocate(h_rm)
        sel = ttnn.reshape(sel, [1, 1, MAX_DECODE_BATCH, self.hidden_size // TP])
        g = self.ccl.all_gather(sel, dim=3, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(sel)
        out = ttnn.to_memory_config(g, self.residual_mem)
        ttnn.deallocate(g)
        return out

    def chunk_plan(self, start_pos, num_tokens):
        """[(chunk_start, real_rows, bucket_rows)] covering positions ``[start_pos, start_pos + num_tokens)``."""
        assert start_pos % self.prefill_chunk == 0, "prefill continuations start on a prefill-chunk boundary"
        assert num_tokens >= 1 and start_pos + num_tokens <= self.max_context
        plan = []
        for c0 in range(start_pos, start_pos + num_tokens, self.prefill_chunk):
            n = min(self.prefill_chunk, start_pos + num_tokens - c0)
            plan.append((c0, n, prefill_bucket(n, self.prefill_chunk)))
        return plan

    def warm_prefill(self, log=print):
        """Compile every prefill program before any trace exists (``{chunk starts} x {buckets}``).

        All bucket lengths at chunk start 0 run through every layer (each layer's matmul / norm / CCL programs are
        keyed by the row count). Programs keyed by the chunk start (RoPE-table and page-table slices, chunked SDPA)
        depend only on the layer kind, so the other chunk starts run through one layer of each kind. Writes go to
        the scratch block only (warmup page table), so no request state is touched.
        """
        table = self.device_ints(torch.full((1, self.pages_per_user), self.scratch_block, dtype=torch.int32))
        rows_idx = self.device_ints(torch.zeros(1, MAX_DECODE_BATCH, dtype=torch.int32), dtype=ttnn.uint32)
        buckets = [b for b in PREFILL_BUCKETS if b <= self.prefill_chunk]
        for b in buckets:
            h = self.prefill_chunk_hidden(torch.zeros(b, dtype=torch.int32), table, 0)
            ttnn.deallocate(self.select_rows(h, rows_idx))
            ttnn.deallocate(h)
        reps = list(self.kind_layer.values())
        local = self.hidden_size // TP
        for c0 in range(self.prefill_chunk, self.max_context, self.prefill_chunk):
            for b in buckets:
                x = ttnn.from_torch(
                    torch.zeros(1, 1, b, local * TP),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.mesh,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ShardTensorToMesh(self.mesh, dim=3),
                )
                for layer in reps:
                    ttnn.deallocate(layer.prefill_forward(x, page_table=table, start_pos=c0, user_id=0))
                ttnn.deallocate(x)
            if c0 % (8 * self.prefill_chunk) == 0:
                log(f"warm prefill: chunk start {c0}")
        ttnn.synchronize_device(self.mesh)
        ttnn.deallocate(table)
        ttnn.deallocate(rows_idx)
