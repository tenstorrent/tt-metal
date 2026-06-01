# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


import torch
import ttnn
from models.experimental.tt_symbiote.core.module import (
    TTNNModule,
    run_on_devices,
    SHARDED_COLLECTIVE_LINEAR_DEVICE_ARCHS,
)
from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor
from models.experimental.tt_symbiote.core.run_config import trace_enabled
from models.experimental.tt_symbiote.modules.attention import (
    TTNNPagedAttentionKVCache,
    TTNNSDPAAttention,
)
from models.experimental.tt_symbiote.modules.linear import (
    TTNNLinearLLamaIColShardedWAllReduced,
    TTNNLinearLLamaIReplicatedWColSharded,
    _tp_mesh_mapper,
    _tp_requires_ccl,
)
from models.experimental.tt_symbiote.modules.rope import BailingRotarySetup

try:
    from transformers.cache_utils import Cache
except ImportError:
    Cache = object


class _TTNNDotsOCRQKVPrefillLinear(TTNNLinearLLamaIColShardedWAllReduced):
    """QKV-shaped linear for the prefill path only.

    The parent class pre-allocates a DRAM_WIDTH_SHARDED copy of the weight
    for the decode fast path whenever in/out features match the canonical
    dots.ocr QKV (K=1536, N=2048). The prefill weight is conventional-order
    ([Q_all|K_all|V_all]) and only ever used at seq_len > 1, where the
    DRAM-sharded matmul kernel doesn't fire anyway — so the second device
    copy (~1.7 MB / layer) would just sit unused. Disable it.
    """

    def _qkv_use_dram_sharded(self) -> bool:
        return False

    def _prefill_matmul_override(self, input_shape):
        # Tuned QKV prefill matmul from test_prefill_ops_sequence_univ_2.py: 8x8 grid, in0_block_w=6,
        # 2D mcast, in0/out L1-interleaved (132 TFLOPs, ~1.61x vs the auto-heuristic). The tiling is
        # specific to M=2816 (88 M-tiles -> per_core_M=11) and N=2048 (64 N-tiles -> per_core_N=8),
        # so it only fires for that bucket; other sequence lengths fall back to the adaptive config.
        m_tiles = (int(input_shape[-2]) + 31) // 32
        n = int(self.tt_weight.shape[-1])
        if m_tiles == 88 and n == 2048:
            return (
                ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
                    in0_block_w=6,
                    out_subblock_h=1,
                    out_subblock_w=8,
                    out_block_h=11,
                    out_block_w=8,
                    per_core_M=11,
                    per_core_N=8,
                    transpose_mcast=False,
                    fused_activation=None,
                    fuse_batch=True,
                ),
                ttnn.L1_MEMORY_CONFIG,
            )
        return None, None


class _TTNNDotsOCROProjPrefillLinear(TTNNLinearLLamaIReplicatedWColSharded):
    """O-projection with a tuned block-sharded prefill matmul (op 14).

    Decode keeps the parent's DRAM-width-sharded fast path unchanged. Prefill
    at the canonical dots.ocr shape (M=2816 -> 88 M-tiles, K=N=1536) runs the
    tuned 8x8 2D-mcast matmul from test_prefill_ops_sequence_univ_2.py op 14:
    BLOCK_SHARDED in0/out on the 8x8 grid, BF16 x BFP4 -> BF16, LoFi
    (~97.6 us, 136 TFLOPs). The tiling (in0_block_w=6, per_core_M=11,
    per_core_N=6) is what the adaptive prefill helper already picks; the win is
    keeping in0 and out BLOCK_SHARDED instead of DRAM-interleaved.

    The 2D-mcast kernel needs a DRAM_INTERLEAVED operand B, while the parent
    stores the weight DRAM_WIDTH_SHARDED for its decode kernel, so a second
    BFP4 DRAM_INTERLEAVED weight copy (~0.6 MB / layer) is built here. Only the
    single-device / pure-DP path is affected (gated on ``not _tp_requires_ccl``,
    like every other fast path); TP keeps the parent's CCL forward. The
    block-sharded output is resharded back to DRAM here so nothing downstream
    changes -- the fused residual+LN (ops 15-16) will later consume it sharded.
    """

    _PREFILL_M_TILES = 88  # M=2816
    _PREFILL_DIM = 1536

    def move_weights_to_device_impl(self):
        # Clone the raw [out, in] torch weight before the parent preprocess
        # consumes/transposes ``tt_weight_host`` into the DRAM-sharded copy.
        raw_weight_torch = self.tt_weight_host.clone() if isinstance(self.tt_weight_host, torch.Tensor) else None
        super().move_weights_to_device_impl()

        self._prefill_weight = None
        self._prefill_in0_mem = None
        self._prefill_out_mem = None
        self._prefill_pc = None

        shape_matches = int(self.in_features) == self._PREFILL_DIM and int(self.out_features) == self._PREFILL_DIM
        if raw_weight_torch is None or _tp_requires_ccl(self.device) or not shape_matches:
            return

        weight_t = raw_weight_torch.T.contiguous()  # [K, N] = [in, out]
        self._prefill_weight = ttnn.as_tensor(
            weight_t,
            device=self.device,
            dtype=getattr(self, "_weight_dtype", ttnn.bfloat4_b),
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=_tp_mesh_mapper(self.device, self.weight_dim),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        m = self._PREFILL_M_TILES * ttnn.TILE_SIZE
        self._prefill_in0_mem = ttnn.create_sharded_memory_config(
            (1, 1, m, self._PREFILL_DIM),
            core_grid=ttnn.CoreGrid(y=8, x=8),
            strategy=ttnn.ShardStrategy.BLOCK,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )
        self._prefill_out_mem = ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            buffer_type=ttnn.BufferType.L1,
        )
        # per_core_M=11 (88/8), per_core_N=6 (48/8), in0_block_w=6 (K=48 tiles / 8 grid rows).
        self._prefill_pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
            in0_block_w=6,
            out_subblock_h=1,
            out_subblock_w=3,
            out_block_h=11,
            out_block_w=6,
            per_core_M=11,
            per_core_N=6,
            transpose_mcast=False,
            fused_activation=None,
        )

    @run_on_devices(*SHARDED_COLLECTIVE_LINEAR_DEVICE_ARCHS)
    def forward(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
        if getattr(self, "_prefill_weight", None) is not None and not _tp_requires_ccl(self.device):
            if input_tensor.layout != ttnn.TILE_LAYOUT:
                input_tensor = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            in_shape = list(input_tensor.shape)
            shape_4d = list(in_shape)
            while len(shape_4d) < 4:
                shape_4d.insert(1, 1)
            m_tiles = (int(shape_4d[-2]) + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE
            if m_tiles == self._PREFILL_M_TILES and int(shape_4d[-1]) == self._PREFILL_DIM:
                x = ttnn.reshape(input_tensor, shape_4d)
                x_bs = ttnn.to_memory_config(x, self._prefill_in0_mem)
                out = ttnn.matmul(
                    x_bs,
                    self._prefill_weight,
                    program_config=self._prefill_pc,
                    memory_config=self._prefill_out_mem,
                    dtype=ttnn.bfloat16,
                    compute_kernel_config=self.compute_kernel_config,
                )
                ttnn.deallocate(x_bs)
                # Leave the output BLOCK_SHARDED on the 8x8 grid: the decoder
                # layer consumes it directly for the block-sharded residual-add
                # + sharded RMSNorm (ops 15-16), so the sharded_to_interleaved
                # that used to sit here is gone.
                return ttnn.reshape(out, in_shape[:-1] + [-1])
        return super().forward(input_tensor)


@trace_enabled
class TTNNDotsOCRAttention(TTNNModule):
    _shared_rotary_setups = {}

    def __init__(self):
        super().__init__()
        self.num_attention_heads = None
        self.num_key_value_heads = None
        self.num_key_value_groups = None
        self.head_dim = None
        self.hidden_size = None
        self.scaling = None
        self.is_causal = True
        self.layer_idx = None
        self.qkv_proj = None
        self.qkv_proj_prefill = None
        self.o_proj = None
        self.sdpa = None
        self.core_grid = None
        self._q_bias_torch = None
        self._k_bias_torch = None
        self._v_bias_torch = None
        self._qkv_bias_torch = None
        self._q_size = None
        self._kv_size = None

    @classmethod
    def from_torch(cls, hf_attn):
        new_attn = cls()
        new_attn._fallback_torch_layer = hf_attn

        config = hf_attn.config
        new_attn.num_attention_heads = config.num_attention_heads
        new_attn.num_key_value_heads = config.num_key_value_heads
        new_attn.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        new_attn.head_dim = config.hidden_size // config.num_attention_heads
        new_attn.hidden_size = config.hidden_size
        new_attn.scaling = getattr(hf_attn, "scaling", new_attn.head_dim**-0.5)
        new_attn.layer_idx = hf_attn.layer_idx

        q_size = new_attn.num_attention_heads * new_attn.head_dim
        kv_size = new_attn.num_key_value_heads * new_attn.head_dim
        new_attn._q_size = q_size
        new_attn._kv_size = kv_size

        # KV-group-interleaved fused-QKV layout (rows along the output-feature
        # dim of the linear). Instead of the conventional
        # ``cat([Q_all, K_all, V_all])`` block layout, we interleave heads by
        # KV group so the fused tensor looks like:
        #   [Q_g0 (num_q_heads/num_kv_heads heads), K_g0 (1 head), V_g0 (1 head),
        #    Q_g1 (...), K_g1, V_g1, ...]
        # This is the *per-shard* layout the Sharded program factory of
        # ``nlp_create_qkv_heads`` reads (see
        # nlp_create_qkv_heads_program_factory.cpp:387-392: each input shard
        # is assumed to be ``[Q_group_i | K_i | V_i]`` in width). Producing
        # the matmul output already in this layout means a single
        # width-shard reshard of ``qkv_states`` engages the Sharded factory
        # in decode (kernel runs across ``num_q_heads`` cores instead of
        # falling back to the 1-core interleaved variant).
        # Prefill keeps using the interleaved create-heads factory, which
        # assumes ``[Q_all, K_all, V_all]``, so prefill uses a separate
        # weight in that conventional order (``qkv_proj_prefill`` below).
        num_q_heads = new_attn.num_attention_heads
        num_kv_heads = new_attn.num_key_value_heads
        num_q_per_kv = num_q_heads // num_kv_heads
        head_dim = new_attn.head_dim
        hidden_dim = new_attn.hidden_size

        q_w_grouped = hf_attn.q_proj.weight.data.view(num_kv_heads, num_q_per_kv, head_dim, hidden_dim)
        k_w_grouped = hf_attn.k_proj.weight.data.view(num_kv_heads, 1, head_dim, hidden_dim)
        v_w_grouped = hf_attn.v_proj.weight.data.view(num_kv_heads, 1, head_dim, hidden_dim)
        fused_weight = torch.cat([q_w_grouped, k_w_grouped, v_w_grouped], dim=1).reshape(-1, hidden_dim)

        # Fuse Q/K/V bias into the qkv linear so it can ride the matmul kernel
        # (1D/2D mcast both fold a passed bias tensor into the post-accumulation
        # step of the matmul kernel). This eliminates one BinaryNg per layer in
        # decode mode without changing numerics.
        q_bias = hf_attn.q_proj.bias.data.clone() if hf_attn.q_proj.bias is not None else None
        k_bias = hf_attn.k_proj.bias.data.clone() if hf_attn.k_proj.bias is not None else None
        v_bias = hf_attn.v_proj.bias.data.clone() if hf_attn.v_proj.bias is not None else None
        new_attn._q_bias_torch = q_bias
        new_attn._k_bias_torch = k_bias
        new_attn._v_bias_torch = v_bias

        has_any_bias = q_bias is not None or k_bias is not None or v_bias is not None
        if has_any_bias:
            zeros_dtype = fused_weight.dtype
            qb = q_bias if q_bias is not None else torch.zeros(q_size, dtype=zeros_dtype)
            kb = k_bias if k_bias is not None else torch.zeros(kv_size, dtype=zeros_dtype)
            vb = v_bias if v_bias is not None else torch.zeros(kv_size, dtype=zeros_dtype)
            # Apply the same KV-group-interleaved permute to the fused bias.
            qb_grouped = qb.view(num_kv_heads, num_q_per_kv, head_dim)
            kb_grouped = kb.view(num_kv_heads, 1, head_dim)
            vb_grouped = vb.view(num_kv_heads, 1, head_dim)
            new_attn._qkv_bias_torch = torch.cat([qb_grouped, kb_grouped, vb_grouped], dim=1).reshape(-1)

        fused_linear = torch.nn.Linear(new_attn.hidden_size, q_size + 2 * kv_size, bias=has_any_bias)
        fused_linear.weight.data = fused_weight
        if has_any_bias:
            fused_linear.bias.data = new_attn._qkv_bias_torch
        new_attn.qkv_proj = TTNNLinearLLamaIColShardedWAllReduced.from_torch(fused_linear)

        # Conventional-order QKV weight ([Q_all | K_all | V_all]) for prefill.
        # Prefill's nlp_create_qkv_heads runs the Interleaved factory which
        # assumes this block layout; storing the weight that way produces a
        # post-matmul tensor that can flow straight into create_heads,
        # eliminating the 6 BF16 slices + concat that _undo_qkv_kv_group_permute
        # used to do every prefill (Tracy: ~25 ms / layer at S=2816).
        # Memory cost: +1 QKV weight per layer (~3 MB BFP8 on dots.ocr,
        # in_features=1536 / out_features=2048).
        fused_weight_conv = torch.cat(
            [hf_attn.q_proj.weight.data, hf_attn.k_proj.weight.data, hf_attn.v_proj.weight.data], dim=0
        )
        fused_linear_conv = torch.nn.Linear(new_attn.hidden_size, q_size + 2 * kv_size, bias=has_any_bias)
        fused_linear_conv.weight.data = fused_weight_conv
        if has_any_bias:
            qb_conv = q_bias if q_bias is not None else torch.zeros(q_size, dtype=fused_weight_conv.dtype)
            kb_conv = k_bias if k_bias is not None else torch.zeros(kv_size, dtype=fused_weight_conv.dtype)
            vb_conv = v_bias if v_bias is not None else torch.zeros(kv_size, dtype=fused_weight_conv.dtype)
            fused_linear_conv.bias.data = torch.cat([qb_conv, kb_conv, vb_conv], dim=0)
        new_attn.qkv_proj_prefill = _TTNNDotsOCRQKVPrefillLinear.from_torch(fused_linear_conv)

        # O projection (block-sharded tuned prefill matmul + decode DRAM-sharded fast path)
        new_attn.o_proj = _TTNNDotsOCROProjPrefillLinear.from_torch(hf_attn.o_proj)

        new_attn.sdpa = TTNNSDPAAttention()
        new_attn.core_grid = ttnn.CoreGrid(y=8, x=8)

        return new_attn

    def move_weights_to_device_impl(self):
        super().move_weights_to_device_impl()

        grid = self.device.compute_with_storage_grid_size()
        self.core_grid = ttnn.CoreGrid(y=grid.y, x=grid.x)

        if self.sdpa.program_config is None:
            # Prefill SDPA: tuned 8x8 grid, q_chunk=256 / k_chunk=256 (both divide M=2816=256*11),
            # exact softmax (exp_approx_mode=False) + HiFi2 (see compute_kernel_config below).
            # This 256/256 schedule matches the gpt_oss prefill and llama3-70b configs and the
            # standalone prefill op-sequence tuning (test_prefill_ops_sequence_univ_2.py).
            self.sdpa.program_config = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(self.core_grid.x, self.core_grid.y),
                q_chunk_size=256,
                k_chunk_size=256,
                exp_approx_mode=False,
            )
            self.sdpa.decode_program_config = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(self.core_grid.x, self.core_grid.y),
                q_chunk_size=0,
                k_chunk_size=0,
                exp_approx_mode=True,
            )
            self.sdpa.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.LoFi,
                math_approx_mode=True,
                fp32_dest_acc_en=False,
                packer_l1_acc=True,
            )
            # Decode SDPA: HiFi2 (was LoFi in commit d1b17d1a3c6 -- swapped back
            # because LoFi at the per-token batch=1 K/V cache reads produces
            # off-by-many-tokens argmax errors visible as garbled output. The
            # earlier "validation" run that approved LoFi was confounded by the
            # broken DRAM-sharded LM head also in that commit, so the LoFi delta
            # was masked. Keep at HiFi2 until a clean A/B confirms it's safe.)
            self.sdpa.decode_compute_kernel_config = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi2,
                math_approx_mode=True,
                fp32_dest_acc_en=False,
                packer_l1_acc=True,
            )

        # Override QKV compute config: HiFi2 for decode
        self.qkv_proj.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

        mesh_mapper = ttnn.ReplicateTensorToMesh(self.device) if self.device.get_num_devices() > 1 else None

        if self.device.get_num_devices() > 1:
            self._decode_cur_pos = ttnn.from_torch(
                torch.zeros(1, dtype=torch.int32),
                device=self.device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=mesh_mapper,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            self._decode_cur_pos = None

        # _q_bias / _k_bias / _v_bias / _qkv_bias are no longer materialized as
        # separate device tensors — bias is folded into qkv_proj.tt_bias and
        # fused into the matmul kernel by the linear layer.

        config = self._fallback_torch_layer.config
        rope_params = getattr(config, "rope_parameters", {}) or {}
        rope_theta = getattr(config, "rope_theta", rope_params.get("rope_theta", 1000000.0))
        setup_key = (id(self.device), self.head_dim, rope_theta, 1.0)
        if setup_key not in TTNNDotsOCRAttention._shared_rotary_setups:
            TTNNDotsOCRAttention._shared_rotary_setups[setup_key] = BailingRotarySetup(
                device=self.device,
                head_dim=self.head_dim,
                max_seq_len=131072,
                rope_theta=rope_theta,
                partial_rotary_factor=1.0,
                rope_convention="half_half",
            )
        self._rotary_setup = TTNNDotsOCRAttention._shared_rotary_setups[setup_key]

    def _get_cur_pos_device_tensor(self, cache_position, batch_size):
        cp = cache_position
        if isinstance(cp, TorchTTNNTensor):
            cp = cp.ttnn_tensor
        if len(cp.shape) > 1:
            total_elems = 1
            for d in cp.shape:
                total_elems *= d
            cp = ttnn.reshape(cp, (total_elems,))
        if cp.shape[0] > batch_size:
            cp = ttnn.slice(cp, [0], [batch_size])
        if self._decode_cur_pos is not None:
            ttnn.copy(cp, self._decode_cur_pos)
            return self._decode_cur_pos
        return cp

    def _project_qkv_fused(self, hidden_states, batch_size, seq_length, proj=None):
        """Project hidden states to fused QKV tensor, ready for nlp_create_qkv_heads.

        Bias (if any) is fused into the qkv_proj matmul kernel, so the output
        of qkv_proj is already (W·x + b). On single-device decode, we only need
        to make sure the result lives in L1 to keep nlp_create_qkv_heads on
        L1 inputs.

        ``proj`` selects which QKV linear to use: defaults to ``self.qkv_proj``
        (KV-group-interleaved layout, decode fast path). Prefill passes
        ``self.qkv_proj_prefill`` for conventional ``[Q_all|K_all|V_all]``
        output that flows straight into the Interleaved create_heads factory.
        """
        if hidden_states.layout != ttnn.TILE_LAYOUT:
            hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)

        if proj is None:
            proj = self.qkv_proj
        qkv_states = proj(hidden_states)

        is_decode = int(seq_length) == 1
        if is_decode and qkv_states.memory_config().buffer_type != ttnn.BufferType.L1:
            qkv_states = ttnn.to_memory_config(qkv_states, ttnn.L1_MEMORY_CONFIG)

        qkv_states = ttnn.reshape(qkv_states, (batch_size, 1, seq_length, -1))

        return qkv_states

    def _forward_prefill(self, hidden_states, attention_mask, past_key_values, cache_position):
        batch_size, seq_length = hidden_states.shape[0], hidden_states.shape[1]

        # Prefill uses qkv_proj_prefill (conventional [Q_all|K_all|V_all] weight),
        # so the matmul output is already in the layout the Interleaved
        # create_heads factory expects. This drops the 6 BF16 slices + concat
        # _undo_qkv_kv_group_permute used to do every prefill (Tracy: ~25 ms
        # / layer at S=2816). Decode keeps using self.qkv_proj (KV-group-
        # interleaved) to engage the Sharded create_heads factory.
        qkv_states = self._project_qkv_fused(hidden_states, batch_size, seq_length, proj=self.qkv_proj_prefill)
        query_states, key_states, value_states = ttnn.experimental.nlp_create_qkv_heads(
            qkv_states,
            num_heads=self.num_attention_heads,
            num_kv_heads=self.num_key_value_heads,
            transpose_k_heads=False,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(qkv_states)

        seq_len = query_states.shape[2]
        cos, sin = self._rotary_setup.get_cos_sin_for_prefill(seq_len)

        # ``ttnn.experimental.rotary_embedding`` preserves input dtype, so
        # Q/K stay BFP8 through the rotary instead of round-tripping
        # through BF16 (saves 2 typecasts per layer in text-decoder
        # prefill, ~0.7 ms/layer at this seq_len).
        query_states = ttnn.experimental.rotary_embedding(query_states, cos, sin)
        key_states = ttnn.experimental.rotary_embedding(key_states, cos, sin)

        if query_states.shape[2] != seq_len:
            query_states = query_states[:, :, :seq_len, :]
        if key_states.shape[2] != seq_len:
            key_states = key_states[:, :, :seq_len, :]

        use_paged = isinstance(past_key_values, TTNNPagedAttentionKVCache)
        if past_key_values is not None and use_paged:
            # ``paged_fill_cache`` requires FLOAT32/BFLOAT16 *input* when the
            # on-device cache is BF16 (see paged_fill_cache_device_operation.cpp).
            # QKV matmul uses BF16×BFP4→BFP8; keep BFP8 for SDPA below but cast
            # only the fill-cache operands to BF16 for the write.
            k_fill = (
                key_states
                if key_states.dtype == ttnn.bfloat16
                else ttnn.typecast(key_states, ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)
            )
            v_fill = (
                value_states
                if value_states.dtype == ttnn.bfloat16
                else ttnn.typecast(value_states, ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)
            )
            past_key_values.paged_fill_on_device(k_fill, v_fill, layer_idx=self.layer_idx, batch_idx=0)
            if k_fill is not key_states:
                ttnn.deallocate(k_fill)
            if v_fill is not value_states:
                ttnn.deallocate(v_fill)

        attn_output = self.sdpa(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0,
            scaling=self.scaling,
            is_causal=self.is_causal,
            transpose_output=False,
        )

        attn_output = ttnn.experimental.nlp_concat_heads(attn_output, memory_config=ttnn.L1_MEMORY_CONFIG)
        attn_output = ttnn.squeeze(attn_output, 1)

        attn_output = self.o_proj(attn_output)
        return attn_output, None

    def _forward_decode_paged(
        self,
        hidden_states,
        attention_mask,
        past_key_values,
        cache_position,
        decode_cur_pos_tt=None,
        decode_cos_sin=None,
    ):
        batch_size, seq_length = hidden_states.shape[0], hidden_states.shape[1]

        if decode_cur_pos_tt is not None:
            cur_pos_tt = decode_cur_pos_tt
        else:
            cur_pos_tt = self._get_cur_pos_device_tensor(cache_position, batch_size)

        qkv_states = self._project_qkv_fused(hidden_states, batch_size, seq_length)

        is_decode = int(seq_length) == 1

        # Engage the multi-core Sharded program factory of
        # ``nlp_create_qkv_heads`` in decode mode by width-sharding the qkv
        # matmul output across ``num_kv_heads`` cores. The fused QKV weight
        # is already laid out as ``[Q_g0, K_0, V_0, Q_g1, K_1, V_1, ...]``
        # (see ``from_torch``) so each width shard matches the per-shard
        # ``[Q_group_i | K_i | V_i]`` layout the kernel reads
        # (see nlp_create_qkv_heads_program_factory.cpp:387-392). With
        # interleaved input the op falls back to its 1-core Interleaved
        # factory; sharding alone (without the matching weight layout)
        # silently corrupts Q/K/V.
        if is_decode:
            num_q_per_kv = self.num_attention_heads // self.num_key_value_heads
            qkv_input_shard_spec = ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(
                            ttnn.CoreCoord(0, 0),
                            ttnn.CoreCoord(self.num_key_value_heads - 1, 0),
                        )
                    ]
                ),
                [ttnn.TILE_SIZE, (num_q_per_kv + 2) * self.head_dim],
                ttnn.ShardOrientation.ROW_MAJOR,
            )
            qkv_input_mem_config = ttnn.MemoryConfig(
                memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                buffer_type=ttnn.BufferType.L1,
                shard_spec=qkv_input_shard_spec,
            )
            qkv_states = ttnn.to_memory_config(qkv_states, qkv_input_mem_config)
            # Sharded factory creates its own per-output shard specs; only
            # the layout/buffer fields of ``memory_config`` matter here.
            nlp_heads_mc = ttnn.MemoryConfig(
                memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                buffer_type=ttnn.BufferType.L1,
                shard_spec=qkv_input_shard_spec,
            )
        else:
            nlp_heads_mc = ttnn.DRAM_MEMORY_CONFIG

        query_states, key_states, value_states = ttnn.experimental.nlp_create_qkv_heads(
            qkv_states,
            num_heads=self.num_attention_heads,
            num_kv_heads=self.num_key_value_heads,
            transpose_k_heads=False,
            memory_config=nlp_heads_mc,
        )
        ttnn.deallocate(qkv_states)

        # V skips rotary, so the Sharded factory's HEIGHT_SHARDED layout
        # would persist into the downstream ``permute(2,0,1,3)`` kernel
        # which has no sharded TensorAccessor path. Reshard back to L1
        # interleaved while Q/K continue sharded through rotary (rotary
        # supports HEIGHT_SHARDED input and reshards to its requested
        # output mem config in the same kernel).
        if is_decode and value_states.is_sharded():
            value_states = ttnn.sharded_to_interleaved(value_states, ttnn.L1_MEMORY_CONFIG)

        # ``typecast(bfloat16)`` on Q/K was conditional on dtype != bfloat16.
        # qkv_proj emits bfloat16 and ``nlp_create_qkv_heads`` preserves dtype,
        # so the condition is always false in this config. Dropping the dead
        # branches removes two ops from the per-token graph.

        if decode_cos_sin is not None:
            cos, sin = decode_cos_sin
        else:
            cos, sin = self._rotary_setup.get_cos_sin_for_decode(cur_pos_tt)
        # K stays in L1 (reshaped/sharded before paged_update_on_device which accepts any
        # memory layout). Q must end up DRAM for paged_sdpa_decode (`Q_memcfg.buffer_type()
        # == DRAM` asserted in sdpa_decode_device_operation.cpp:89). Since rotary_embedding
        # defaults its output memory_config to the input's, we explicitly request DRAM
        # for Q because nlp_create_qkv_heads now produces L1 in decode.
        query_states = ttnn.experimental.rotary_embedding(query_states, cos, sin, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        key_states = ttnn.experimental.rotary_embedding(key_states, cos, sin, memory_config=ttnn.L1_MEMORY_CONFIG)

        if query_states.shape[2] != seq_length:
            query_states = query_states[:, :, :seq_length, :]
        if key_states.shape[2] != seq_length:
            key_states = ttnn.slice(
                key_states,
                [0, 0, 0, 0],
                [int(key_states.shape[0]), int(key_states.shape[1]), int(seq_length), int(key_states.shape[3])],
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )

        # Permute [B, H, S, D] -> [S, B, H, D] for paged attention kernels
        query_states = ttnn.permute(query_states, (2, 0, 1, 3))
        kv_key = ttnn.permute(key_states, (2, 0, 1, 3))
        kv_value = ttnn.permute(value_states, (2, 0, 1, 3))

        # ``typecast(bfloat16)`` on kv_value was conditional on dtype != bfloat16
        # and never triggers in this config (nlp_create_qkv_heads -> permute
        # preserves bfloat16). Dropping the dead branch.

        tile_size = 32
        shard_h = ((self.num_key_value_heads + tile_size - 1) // tile_size) * tile_size
        shard_cfg = ttnn.create_sharded_memory_config(
            shape=(shard_h, self.head_dim),
            core_grid=ttnn.CoreGrid(y=1, x=batch_size),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )
        kv_key = ttnn.to_memory_config(kv_key, shard_cfg)
        kv_value = ttnn.to_memory_config(kv_value, shard_cfg)

        past_key_values.paged_update_on_device(kv_key, kv_value, layer_idx=self.layer_idx, current_pos=cur_pos_tt)
        ttnn.deallocate(kv_key)
        ttnn.deallocate(kv_value)

        attn_output = past_key_values.paged_sdpa_decode(
            query_states,
            self.layer_idx,
            current_pos=cur_pos_tt,
            scale=self.scaling,
            program_config=getattr(self.sdpa, "decode_program_config", self.sdpa.program_config),
            compute_kernel_config=getattr(self.sdpa, "decode_compute_kernel_config", self.sdpa.compute_kernel_config),
        )

        sdpa_output_memcfg = ttnn.create_sharded_memory_config(
            shape=(32, self.head_dim),
            core_grid=ttnn.CoreGrid(y=1, x=batch_size),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        attn_output = ttnn.to_memory_config(attn_output, sdpa_output_memcfg)
        attn_output = ttnn.experimental.nlp_concat_heads_decode(
            attn_output,
            num_heads=self.num_attention_heads,
        )
        if batch_size < 32:
            attn_output = ttnn.to_memory_config(attn_output, ttnn.L1_MEMORY_CONFIG)
            attn_output = ttnn.slice(attn_output, [0, 0, 0, 0], [1, 1, batch_size, int(attn_output.shape[-1])])

        attn_output = self.o_proj(attn_output)
        attn_output = ttnn.squeeze(attn_output, 1)
        return attn_output, None

    def forward(
        self,
        hidden_states,
        position_embeddings=None,
        attention_mask=None,
        past_key_values=None,
        cache_position=None,
        position_ids=None,
        **kwargs,
    ):
        seq_length = hidden_states.shape[1]
        use_paged = isinstance(past_key_values, TTNNPagedAttentionKVCache)

        if use_paged and seq_length == 1:
            return self._forward_decode_paged(
                hidden_states,
                attention_mask,
                past_key_values,
                cache_position,
                decode_cur_pos_tt=kwargs.get("decode_cur_pos_tt"),
                decode_cos_sin=kwargs.get("decode_cos_sin"),
            )
        else:
            return self._forward_prefill(hidden_states, attention_mask, past_key_values, cache_position)
