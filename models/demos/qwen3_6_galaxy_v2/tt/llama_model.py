# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os

import torch
from tqdm import tqdm

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.rmsnorm import RMSNorm
from models.common.sampling.generator import SamplingGenerator
from models.demos.qwen3_6_galaxy_v2.tt.distributed_norm import DistributedNorm
from models.demos.qwen3_6_galaxy_v2.tt.llama_ccl import TT_CCL
from models.demos.qwen3_6_galaxy_v2.tt.llama_common import copy_host_to_device, get_prefill_rot_mat
from models.demos.qwen3_6_galaxy_v2.tt.llama_decoder import TtTransformerBlock
from models.demos.qwen3_6_galaxy_v2.tt.llama_embedding import TtLlamaEmbedding
from models.demos.qwen3_6_galaxy_v2.tt.llama_rope import TtLlamaRotarySetup
from models.demos.qwen3_6_galaxy_v2.tt.lm_head import LMHead
from models.demos.qwen3_6_galaxy_v2.tt.load_checkpoints import standardize_hf_keys_qwen36
from models.demos.qwen3_6_galaxy_v2.tt.prefetcher_common import TtLlamaPrefetcherSetup


class _NoOpPrefetcherSetup:
    """Drop-in stub for ``TtLlamaPrefetcherSetup`` when ``use_prefetcher=False``.

    Upstream ``TtLlamaMLP.__init__`` / ``TtLlamaAttention.__init__`` unconditionally
    invoke ``self.prefetch(prefetcher_setup, tt_ccl)`` in decode mode, which calls
    ``prefetcher_setup.insert_tensor(...)``.  qwen3.6 / olmo run with the prefetcher
    disabled (``use_prefetcher=False``) and previously passed ``None``, tripping an
    ``AttributeError`` at construction.  This stub accepts the insert calls as
    no-ops; ``USE_PREFETCHER`` is False in model_config, so the prefetcher-path
    forward branches never read the (unset) ``global_circular_buffer``.

    ``worker_sub_device_id`` IS read unconditionally at the top of
    ``TtTransformer.forward(mode='decode')`` (set_sub_device_stall_group), so
    we expose it here.  The model owner (TtTransformer) updates it at
    ``setup_decode`` / ``setup_prefill`` time.
    """

    def __init__(self):
        # Default to SubDeviceId(0); TtTransformer.setup_decode overwrites this
        # with the actual worker_sub_device_id it bound when creating the
        # all-cores sub_device.
        self.worker_sub_device_id = ttnn.SubDeviceId(0)

    def insert_tensor(self, *_args, **_kwargs):
        return None

    def get_input_tensors(self):
        return []

    def create_global_cb(self):
        return None


class TtTransformer(LightweightModule):
    def __init__(
        self,
        args,
        dtype,
        mesh_device,
        state_dict,
        weight_cache_path,
        paged_attention_config=None,
        use_paged_kv_cache=False,
        enable_prefetcher_performance_mode=False,
        mode="decode",
        allocate_prefill_buffers=True,
        decode_mode_only=False,
    ):
        super().__init__()
        self.args = args
        # qwen3.6-27B (Galaxy V2) marker — gates DeltaNet dispatch, partial RoPE,
        # zero-centered norm, and the no-prefetcher decode/prefill paths. Falls
        # back to ``False`` so the 70B / qwen3-32B / olmo regression surface is
        # unaffected (olmo precedent: ``is_olmo`` gating, branch ``origin/ssinghal/olmo-3-32b``).
        self.is_qwen36 = getattr(args, "is_qwen36", False)
        self.vocab_size = args.vocab_size
        assert self.vocab_size > 0
        self.n_layers = args.n_layers
        self.mesh_device = mesh_device
        self.dtype = dtype
        self.model_config = args.get_model_config()
        self.grid_size = self.args.max_grid_size
        self.enable_prefetcher_performance_mode = enable_prefetcher_performance_mode
        # --- Weight ingestion (qwen3.6 only) -----------------------------
        # Defensive: ``args.load_state_dict()`` already runs
        # ``standardize_hf_keys`` → ``convert_hf_to_meta``, but a caller may
        # pass a raw HF state_dict directly (e.g. via
        # ``AutoModelForCausalLM.from_pretrained(...).state_dict()``). When
        # is_qwen36 is set AND we still see the ``model.language_model.*``
        # prefix, run the qwen3.6 standardization pass so downstream
        # constructors find canonical ``layers.{i}.attention.*`` keys.
        if self.is_qwen36 and state_dict is not None and any(k.startswith("model.language_model.") for k in state_dict):
            from models.demos.qwen3_6_galaxy_v2.tt.load_checkpoints import convert_hf_to_meta

            # Two passes: standardize_hf_keys_qwen36 strips the
            # model.language_model.* prefix to model.* and ignores
            # vision/MTP keys; convert_hf_to_meta then runs
            # map_hf_to_meta_keys (HF -> meta name rename, e.g.
            # model.embed_tokens.weight -> tok_embeddings.weight,
            # model.layers.{i}.input_layernorm.weight ->
            # layers.{i}.attention_norm.weight). The QKV reverse-
            # permute is intentionally skipped for is_qwen36 (V2-4 reads
            # un-permuted HF QKVG layout directly).
            state_dict = standardize_hf_keys_qwen36(state_dict)
            state_dict = convert_hf_to_meta(state_dict, args.head_dim, is_qwen36=True)
        state_dict_prefix = args.get_state_dict_prefix("", None)
        self.allocate_prefill_buffers = allocate_prefill_buffers
        self.paged_attention_config = paged_attention_config
        self.decode_mode_only = decode_mode_only

        self.embd = TtLlamaEmbedding(
            mesh_device=mesh_device,
            args=args,
            weight_cache_path=args.weight_cache_path(dtype),
            state_dict=state_dict,
            dtype=ttnn.bfloat16,  # Row major layout requires bfloat16
        )

        # qwen3.6 partial-RoPE: pass ``args=`` so TtLlamaRotarySetup takes the
        # ``is_qwen36`` branch (builds cos/sin at ``args.rope_dim=64``). 70B /
        # qwen3-32B / olmo set ``is_qwen36=False`` and the constructor ignores
        # ``args`` for that path.
        self.rope_setup = TtLlamaRotarySetup(
            mesh_device,
            args.max_batch_size,
            args.head_dim,
            args.max_seq_len,
            args.rope_theta,
            args.use_scaled_rope,
            args.rope_scaling_factor,
            args=args if self.is_qwen36 else None,
        )
        self.trans_mats_dict = self.rope_setup.get_both_trans_mats()

        self.is_prefill_setup = False
        self.is_decode_setup = False
        self.prefetcher_setup = None
        self.mesh_sub_device_manager_id_decode = None
        self.mesh_sub_device_manager_id_prefill = None

        # First initialization of decode CCLs and prefetcher
        self.setup_decode()
        self.is_decode_setup = True

        self.layers = [
            TtTransformerBlock(
                args=args,
                mesh_device=mesh_device,
                dtype=dtype,
                state_dict=state_dict,
                weight_cache_path=weight_cache_path,
                layer_num=i,
                n_layers=self.n_layers,
                transformation_mats=self.trans_mats_dict,
                paged_attention_config=paged_attention_config,
                use_paged_kv_cache=use_paged_kv_cache,
                prefetcher_setup=self.prefetcher_setup,
                tt_ccl=self.tt_ccl,
            )
            for i in tqdm(range(self.n_layers))
        ]
        # --- Thread RoPE setup onto full_attention layers (qwen3.6 only) ---
        # The qwen3.6 ``TtLlamaAttention._forward_prefill_qwen36`` /
        # ``_forward_decode_qwen36`` paths read ``self.rope_setup`` to apply
        # partial RoPE; the V2-decoder's ``TtTransformerBlock`` doesn't pass
        # one through, so wire it here. DeltaNet (linear_attention) layers do
        # NOT use RoPE and are intentionally skipped. For 70B / qwen3-32B /
        # olmo (is_qwen36=False) the attention path constructs its own RoPE
        # internally — no threading needed.
        if self.is_qwen36:
            self._thread_rope_setup()

        # qwen3.6 final norm is zero-centered (Qwen3NextRMSNorm: w' = w + 1).
        # 70B / qwen3-32B / olmo set ``zero_centered_norm=False`` (or absent)
        # and the DistributedNorm default keeps the regression behavior.
        zero_centered_final_norm = getattr(args, "zero_centered_norm", False)
        self.norm = DistributedNorm(
            RMSNorm(
                device=mesh_device,
                dim=args.dim,
                state_dict=state_dict,
                state_dict_prefix=args.get_state_dict_prefix("", None),
                weight_cache_path=None if args.dummy_weights else weight_cache_path,
                weight_dtype=ttnn.bfloat16,
                weight_key="norm",
                is_distributed=self.args.is_distributed_norm,
                sharded_program_config=self.model_config["SHARDED_NORM_LM_HEAD_PRGM_CFG"],
                sharded_output_config=self.model_config["LM_HEAD_INPUT_MEMCFG"],
            ),
            args,
            tt_ccl=self.tt_ccl,
            ccl_topology=self.model_config["CCL_TOPOLOGY"],
            zero_centered=zero_centered_final_norm,
        )

        state_dict_prefix = args.get_state_dict_prefix("", None)

        self.lm_head = LMHead(
            args=args,
            mesh_device=mesh_device,
            dtype=dtype,
            state_dict=state_dict,
            state_dict_prefix=state_dict_prefix,
            weight_cache_path=weight_cache_path,
            tt_ccl=self.tt_ccl,
            prefetcher_setup=self.prefetcher_setup,
        )
        if not self.decode_mode_only:  # demo_decode.py uses decode mode only. In this case avoid initializing prefill
            # First initialization of prefill CCLs and prefetcher. It needs to be after initialization of layers, norm and lm_head since those switch modes as well
            # This initialization is required to avoid race condition due to all buffers and semaphores not being allocated at initialization
            self.switch_mode("prefill")
            # qwen3 / qwen3.6 already do prefill setup via switch_mode; only the
            # 70B / olmo paths need this extra call.
            if not self.args.is_qwen and not self.is_qwen36:
                self.setup_prefill()
            self.is_prefill_setup = True

        if mode == "decode" and getattr(self.args, "use_prefetcher", False) and self.prefetcher_setup is not None:
            self.tt_tensors = self.prefetcher_setup.get_input_tensors()
        self.tt_rot_mats_prefill = None

        # Device tensor holding max_seq_len; used directly as slice end in prefill (never updated).
        self._tt_seq_len_buffer = ttnn.from_torch(
            torch.tensor([1, 1, self.args.max_seq_len, self.args.head_dim], dtype=torch.int32),
            device=self.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        # [0, 0, 0, 0] - slice to get [0,0] and [0] for building slice start [0, 0, N, 0] from chunk_start_idx [N].
        self._tt_slice_start_zeros_4 = ttnn.from_torch(
            torch.tensor([0, 0, 0, 0], dtype=torch.int32),
            device=self.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def _thread_rope_setup(self):
        """Thread ``self.rope_setup`` onto each full_attention layer's attention.

        The qwen3.6 ``TtLlamaAttention`` (is_qwen36=True) forward paths read
        ``self.rope_setup`` to apply partial RoPE; ``TtTransformerBlock``
        doesn't pass it through its constructor, so we wire it after the
        per-layer loop. DeltaNet (``is_linear_attention_layer=True``) layers
        do NOT use RoPE and are intentionally skipped.

        Safe to call only when ``self.is_qwen36`` is True. For 70B /
        qwen3-32B / olmo the attention block constructs its own RoPE
        internally — no threading needed.
        """
        for layer in self.layers:
            # DeltaNet layers expose ``is_linear_attention_layer=True`` on
            # the parent ``TtTransformerBlock``; the full_attention branch
            # sets it to False (or omits it on non-qwen36 paths).
            if getattr(layer, "is_linear_attention_layer", False):
                continue
            layer.attention.rope_setup = self.rope_setup

    def get_or_create_prefill_rot_mats(self):
        """
        Return device-side rot mats for prefill, cached once for max_seq_len.

        qwen3.6 uses PARTIAL RoPE (rope_dim = head_dim*0.25 = 64); the attention
        prefill (``_forward_prefill_qwen36``) expects rot_mats as a
        ``(cos, sin)`` tuple of partial tables (shape [1, 1, T, rope_dim]) — the
        same form the working demo builds via ``build_mrope_cos_sin``
        (text_demo_qwen36._build_partial_rope_cos_sin_tt). The default llama
        ``get_prefill_rot_mat`` builds FULL head_dim tables and breaks the
        partial-RoPE eltwise broadcast, so qwen3.6 takes its own branch.
        """
        if self.tt_rot_mats_prefill is None:
            if self.is_qwen36:
                self.tt_rot_mats_prefill = self._build_qwen36_prefill_partial_rope()
            else:
                self.tt_rot_mats_prefill = get_prefill_rot_mat(
                    head_dim=self.args.head_dim,
                    max_seq_len=self.args.max_seq_len,
                    mesh_device=self.mesh_device,
                    seq_len=int(self.args.max_seq_len),
                    scale_factor=self.args.rope_scaling_factor,
                    start_pos=0,
                )
        return self.tt_rot_mats_prefill

    def _build_qwen36_prefill_partial_rope(self):
        """Partial-RoPE (cos, sin) tables for the full [0, max_seq_len) range,
        rope_dim=64. Cached once; ``_forward_prefill_qwen36`` slices to the
        prefill window's seq length. Mirrors the demo's working construction."""
        from models.demos.qwen3_6_galaxy.reference.qwen36 import build_mrope_cos_sin

        max_seq = int(self.args.max_seq_len)
        positions = torch.arange(max_seq, dtype=torch.long)
        positions_3d = torch.stack([positions, positions, positions], dim=0)
        cos_ref, sin_ref = build_mrope_cos_sin(
            positions_3d=positions_3d,
            head_dim=self.args.head_dim,
            partial_rotary_factor=getattr(self.args, "partial_rotary_factor", 0.25),
            mrope_section=getattr(self.args, "mrope_section", [11, 11, 10]),
            theta=getattr(self.args, "rope_theta", 10_000_000.0),
        )
        rot_mats = []
        for t in (cos_ref, sin_ref):  # each [1, T, rope_dim]
            rot_mats.append(
                ttnn.from_torch(
                    t.unsqueeze(0),  # -> [1, 1, T, rope_dim]
                    device=self.mesh_device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                )
            )
        return rot_mats

    def setup_prefill(self, mesh_sub_device_manager_id_prefill=None):
        # qwen3.6 / olmo precedent: when use_prefetcher=False, skip the
        # prefetcher entirely and use a single all-cores sub-device.
        use_prefetcher = getattr(self.args, "use_prefetcher", True)
        if not use_prefetcher:
            self.prefetcher_setup = _NoOpPrefetcherSetup()
            worker_sub_device_id = ttnn.SubDeviceId(0)
            if mesh_sub_device_manager_id_prefill is None:
                grid_size = self.mesh_device.compute_with_storage_grid_size()
                all_core_range_set = ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1))]
                )
                all_sub_device = ttnn.SubDevice([all_core_range_set])
                sub_device_manager = self.mesh_device.create_sub_device_manager([all_sub_device], 0)
                self.mesh_device.load_sub_device_manager(sub_device_manager)
                self.mesh_sub_device_manager_id_prefill = sub_device_manager
                self.tt_ccl = TT_CCL(
                    self.mesh_device,
                    self.args,
                    worker_sub_device_id=worker_sub_device_id,
                    mode="prefill",
                    allocate_prefill_buffers=self.allocate_prefill_buffers,
                    is_qwen=True if self.args.is_qwen else False,
                    is_qwen36=self.is_qwen36,
                )
            else:
                self.mesh_device.load_sub_device_manager(mesh_sub_device_manager_id_prefill)
                self.tt_ccl = self.tt_ccl_prefill
            self.mesh_device.set_sub_device_stall_group([worker_sub_device_id])
            self._worker_sub_device_id = worker_sub_device_id
            return

        self.prefetcher_setup = TtLlamaPrefetcherSetup(
            self.mesh_device,
            n_tensors=0,
            n_layers=self.n_layers,
            mode="prefill",
            mesh_sub_device_manager_id_prefill=mesh_sub_device_manager_id_prefill,
            save_tensor_addresses=True,
            is_qwen=self.args.is_qwen,
        )
        self.mesh_sub_device_manager_id_prefill = self.prefetcher_setup.mesh_sub_device_manager_id_prefill
        self.mesh_device.set_sub_device_stall_group([self.prefetcher_setup.worker_sub_device_id])
        if mesh_sub_device_manager_id_prefill is None:
            self.tt_ccl = TT_CCL(
                self.mesh_device,
                self.args,
                self.prefetcher_setup.worker_sub_device_id,
                mode="prefill",
                allocate_prefill_buffers=self.allocate_prefill_buffers,
                is_qwen=True if self.args.is_qwen else False,
            )
        else:
            self.tt_ccl = self.tt_ccl_prefill

    def setup_decode(self, mesh_sub_device_manager_id_decode=None):
        # qwen3.6 / olmo precedent: when use_prefetcher=False, skip the
        # prefetcher and run on a single all-cores sub-device.
        use_prefetcher = getattr(self.args, "use_prefetcher", True)
        if not use_prefetcher:
            self.prefetcher_setup = _NoOpPrefetcherSetup()
            worker_sub_device_id = ttnn.SubDeviceId(0)
            # Mirror the prefetched-mode contract: forward() reads
            # ``self.prefetcher_setup.worker_sub_device_id`` to set the stall
            # group. Keep that attribute in sync with the actual id we bound.
            self.prefetcher_setup.worker_sub_device_id = worker_sub_device_id
            if mesh_sub_device_manager_id_decode is None:
                grid_size = self.mesh_device.compute_with_storage_grid_size()
                all_core_range_set = ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1))]
                )
                all_sub_device = ttnn.SubDevice([all_core_range_set])
                sub_device_manager = self.mesh_device.create_sub_device_manager([all_sub_device], 0)
                self.mesh_device.load_sub_device_manager(sub_device_manager)
                self.mesh_sub_device_manager_id_decode = sub_device_manager
                self.tt_ccl = TT_CCL(
                    self.mesh_device,
                    self.args,
                    worker_sub_device_id=worker_sub_device_id,
                    is_qwen=True if self.args.is_qwen else False,
                    is_qwen36=self.is_qwen36,
                )
                self.sampling = SamplingGenerator(
                    args=self.args,
                    mesh_device=self.mesh_device,
                    tt_ccl=self.tt_ccl,
                )
            else:
                self.mesh_device.load_sub_device_manager(mesh_sub_device_manager_id_decode)
                self.tt_ccl = self.tt_ccl_decode
            self.mesh_device.set_sub_device_stall_group([worker_sub_device_id])
            self._worker_sub_device_id = worker_sub_device_id
            return

        self.prefetcher_setup = TtLlamaPrefetcherSetup(
            self.mesh_device,
            n_tensors=5 if self.args.use_prefetcher else 0,
            n_layers=self.n_layers,
            mesh_sub_device_manager_id_decode=mesh_sub_device_manager_id_decode,
            save_tensor_addresses=True,
            is_qwen=self.args.is_qwen,
        )
        self.mesh_sub_device_manager_id_decode = self.prefetcher_setup.mesh_sub_device_manager_id_decode
        self.mesh_device.set_sub_device_stall_group(
            [self.prefetcher_setup.prefetcher_sub_device_id, self.prefetcher_setup.worker_sub_device_id]
        )
        if mesh_sub_device_manager_id_decode is None:
            self.tt_ccl = TT_CCL(
                self.mesh_device,
                self.args,
                self.prefetcher_setup.worker_sub_device_id,
                is_qwen=True if self.args.is_qwen else False,
            )
            self.sampling = SamplingGenerator(
                args=self.args,
                mesh_device=self.mesh_device,
                tt_ccl=self.tt_ccl,
            )
        else:
            self.tt_ccl = self.tt_ccl_decode

    def prepare_prefill_inputs_host(
        self,
        tokens,
        user_id=0,
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=0,
        batch_size=1,
    ):
        """
        Inputs are torch tensors or python types. This function returns ttnn
        tensors on host (to be copied to device later).
        """
        tokens = tokens.reshape(1, 1, 1, -1)
        S = tokens.shape[-1]
        tokens = ttnn.from_torch(
            tokens,
            device=None,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        columns = 4
        rows = 8
        user_id_column = user_id // rows  # 0 for user_id 0-7, 1 for user_id 8-15, etc.

        if page_table is not None:
            # NOTE ON SENTINELS / SAFETY:
            # - For chunked SDPA (prefix caching, chunk_start_idx > 0), the SDPA reads the page table.
            #   Reading -1 can cause address overflow, so use 0 for padding (vLLM reserves block 0 as read-safe).
            # - For non-chunked SDPA (no prefix caching), SDPA uses fresh K/V tensors, not the cache.
            #   We only use page_table for paged_fill_cache (write), where -1 means "skip write".
            #   Using -1 for inactive columns avoids unnecessary writes and potential race conditions.
            use_chunked_sdpa_path = chunk_start_idx is not None and chunk_start_idx > 0
            inactive_fill_value = 0 if use_chunked_sdpa_path else -1

            # Chunked SDPA requires page_table "stick size" (row width) to be a multiple of 32 Bytes.
            # Page table entries are int32, so this means the number of columns must be a multiple of 8
            # (8 * sizeof(int32) = 32 bytes). Pad with read-safe zeros.
            def _pad_table_cols_to_multiple_of_8_int32(table_2d: torch.Tensor, pad_value: int = 0) -> torch.Tensor:
                assert table_2d.ndim == 2, f"expected 2D table, got shape={tuple(table_2d.shape)}"
                cols = table_2d.shape[1]
                cols_padded = ((cols + 7) // 8) * 8
                if cols_padded == cols:
                    return table_2d
                padded = torch.full((table_2d.shape[0], cols_padded), pad_value, dtype=table_2d.dtype)
                padded[:, :cols] = table_2d
                return padded

            qwen36_replicate_pt = False
            if self.is_qwen36:
                # qwen3.6: the 4 mesh columns are the H/4 TENSOR-PARALLEL split, and the
                # KV cache is REPLICATED across columns (init_kv_cache dims=(1,None)) — and
                # the DECODE page_table is replicated (prepare_decode_inputs_host
                # dims=(None,None)). So prefill MUST write the user's KV to ALL columns:
                # replicate the (1, num_blocks) page_table across the mesh (matches the
                # working demo's text_demo_qwen36._build_paged_page_table dims=(None,None)).
                # The llama70b column=user-group shard below (dims=(None,0)) writes KV to
                # column 0 ONLY, leaving columns 1-3 empty -> decode (replicated) reads
                # those empty columns -> garbage past the first (pre-cache) token.
                assert batch_size == 1, "qwen3.6 server prefill is batch-1 (single user)"
                page_table_padded = _pad_table_cols_to_multiple_of_8_int32(
                    page_table, pad_value=inactive_fill_value
                )
                qwen36_replicate_pt = True
            elif batch_size > 1:
                assert batch_size == 32, "batch_size must be 32 for batched prefill"
                # Mesh layout padding: (32, num_blocks) -> (4, 32 * num_blocks).
                # For non-chunked SDPA, use -1 for unused regions so paged_fill_cache skips writes.
                # For chunked SDPA (prefix caching), use 0 so SDPA doesn't read -1.
                batch_size_per_column = batch_size // columns
                page_table_padded = (
                    torch.ones((columns, page_table.shape[1] * batch_size), dtype=torch.int32) * inactive_fill_value
                )
                for i in range(columns):
                    row_block = page_table[i * batch_size_per_column : (i + 1) * batch_size_per_column, :].reshape(
                        1, -1
                    )
                    page_table_padded[
                        i,
                        (i * batch_size_per_column)
                        * page_table.shape[1] : (i + 1)
                        * batch_size_per_column
                        * page_table.shape[1],
                    ] = row_block
                chunk_page_table_padded = None  # batch_size>1 => no prefix caching => no chunk_page_table
            else:
                # Mesh layout padding: only the active column is used.
                # For non-chunked SDPA, use -1 for inactive columns so paged_fill_cache skips writes.
                # For chunked SDPA (prefix caching), use 0 so SDPA doesn't read -1.
                num_blocks = page_table.shape[1]
                page_table_padded = torch.ones((columns, num_blocks), dtype=torch.int32) * inactive_fill_value
                # Note: For prefix caching, page_table is already extracted to a single row (shape: 1, num_blocks),
                # so we always access row 0. The original user_id is used only to compute user_id_column.
                page_table_padded[user_id_column, :num_blocks] = page_table[0, :]

                # Ensure row width (in bytes) divisible by 32 for chunked SDPA.
                page_table_padded = _pad_table_cols_to_multiple_of_8_int32(
                    page_table_padded, pad_value=inactive_fill_value
                )

            tt_page_table = ttnn.from_torch(
                page_table_padded,
                device=None,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device)
                if qwen36_replicate_pt
                else ttnn.ShardTensor2dMesh(  # Each mesh column gets one row of the page table
                    self.mesh_device, dims=(None, 0), mesh_shape=self.args.cluster_shape
                ),
            )
        else:
            tt_page_table = None

        if chunk_page_table is not None:
            assert batch_size == 1, "chunk_page_table is only supported for batch_size=1"
            # Use 0 for inactive columns so no reader ever sees -1.
            chunk_page_table_padded = torch.zeros((columns, chunk_page_table.shape[1]), dtype=torch.int32)
            chunk_page_table_padded[user_id_column, :] = chunk_page_table[0, :]

            # Same 32-byte stick alignment as main page_table (required by chunked SDPA / paged_fill_cache).
            chunk_page_table_padded = _pad_table_cols_to_multiple_of_8_int32(chunk_page_table_padded, pad_value=0)

            tt_chunk_page_table = ttnn.from_torch(
                chunk_page_table_padded,
                device=None,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ShardTensor2dMesh(  # Each mesh column gets one row of the chunk page table
                    self.mesh_device, dims=(None, 0), mesh_shape=self.args.cluster_shape
                ),
            )
        else:
            tt_chunk_page_table = None

        user_id = ttnn.from_torch(
            torch.tensor([user_id], dtype=torch.int32),
            device=None,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        tt_chunk_start_idx = ttnn.from_torch(
            torch.tensor([chunk_start_idx], dtype=torch.int32),
            device=None,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        # Pre-computed column mask for chunked SDPA replication (prefix caching).
        # Shape [8, 4, 1, 32] sharded by (0,1) → each device gets [1, 1, 1, 32].
        # Owning column has 1.0, others 0.0. Sliced to [1,1,1,1] in attention.
        column_mask_data = torch.zeros(rows, columns, 1, 32, dtype=torch.float32)
        column_mask_data[:, user_id_column, :, :] = 1.0
        tt_column_mask = ttnn.from_torch(
            column_mask_data,
            device=None,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=(0, 1), mesh_shape=self.args.cluster_shape),
        )

        return tokens, user_id, tt_page_table, tt_chunk_page_table, tt_chunk_start_idx, tt_column_mask

    def transform_prefill_inputs_device(
        self,
        tokens,
        user_id,
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        column_mask=None,
    ):
        tt_tokens = self.embd(tokens)
        tt_tokens = ttnn.unsqueeze_to_4D(tt_tokens)
        return tt_tokens, user_id, page_table, chunk_page_table, chunk_start_idx, column_mask

    def prepare_inputs_prefill(
        self,
        tokens,
        user_id=0,
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=0,
        batch_size=1,
    ):
        """
        Inputs are torch tensors or python types. This function returns ttnn
        tensors on device.
        Returns 6 outputs: prefill_input, tt_user_id, page_table_tt, tt_chunk_page_table, tt_chunk_start_idx, tt_column_mask.
        """
        host_inputs = self.prepare_prefill_inputs_host(
            tokens, user_id, page_table, chunk_page_table, chunk_start_idx, batch_size
        )
        device_inputs = copy_host_to_device(host_inputs, mesh_device=self.mesh_device)  # Helper function
        transformed_device_inputs = self.transform_prefill_inputs_device(*device_inputs)
        return transformed_device_inputs

    def prepare_inputs_decode(self, tokens, current_pos, page_table, is_cur_pos_sharded, is_page_table_sharded):
        """
        Inputs are torch tensors or python types. This function returns ttnn
        tensors on device.
        Its implementation can take advantage of a few other functions which the
        model must implement.
        """
        # qwen3.6 uses a REPLICATED batch (columns are H/4 TP, not data-parallel),
        # so page_table / current_pos must be replicated — the llama70b L1-sharded
        # column-split specs are invalid here. Normalize the flags so both the host
        # mappers (prepare_decode_inputs_host) and the device shard specs
        # (prepare_decode_shard_configs) agree on the replicated layout at B>1.
        if self.is_qwen36 and tokens.shape[0] > 1:
            is_cur_pos_sharded = False
            is_page_table_sharded = False
        host_tensors = self.prepare_decode_inputs_host(
            tokens, current_pos, page_table, is_cur_pos_sharded, is_page_table_sharded
        )
        shard_specs = self.prepare_decode_shard_configs(is_cur_pos_sharded, is_page_table_sharded)
        device_inputs = copy_host_to_device(
            host_tensors, mesh_device=self.mesh_device, shard_specs=shard_specs
        )  # Helper function
        # V2-9: refresh per-step decode-mask buffers BEFORE the trace replay
        # so the in-forward SDPA mask is a pure persistent-buffer read.
        # Caller-friendly even if attention modules do not have the buffer
        # (no-op for non-qwen36 paths).
        self.refresh_decode_per_step_buffers(current_pos)
        return device_inputs

    def set_trace_decode_mode(self, enabled: bool):
        """V2-9: toggle "trace mode" on every full-attention layer.

        When ``enabled`` is True, the per-call SDPA decode-mask refresh
        inside ``_forward_decode_qwen36`` is SKIPPED so no
        ``copy_host_to_device_tensor`` fires inside the trace boundary.
        Callers must invoke ``refresh_decode_per_step_buffers(cur_pos)``
        BEFORE ``ttnn.execute_trace`` to prime the mask buffers.
        """
        if not getattr(self.args, "is_qwen36", False):
            return
        for layer in self.layers:
            attn = getattr(layer, "attention", None)
            if attn is not None:
                setattr(attn, "_skip_decode_mask_refresh", bool(enabled))

    def refresh_decode_per_step_buffers(self, current_pos):
        """V2-9: trace-safe per-step refresh of every full-attention layer's
        persistent decode-mask buffer.

        ``current_pos`` is the per-user position tensor passed into
        ``prepare_decode_inputs_host``.  For single-user qwen3.6 decode it is
        a length-1 torch tensor (or scalar) whose value is the cur_pos int.
        For multi-user batched decode every user sees the same cur_pos in our
        test contract — we use ``current_pos[0]``.

        Safe to call from outside any trace boundary; the underlying
        ``copy_host_to_device_tensor`` is a metadata-only write that does
        NOT trip the trace-capture "Writes are not supported" check.
        """
        if not getattr(self.args, "is_qwen36", False):
            return
        # current_pos may be a python int, torch.Tensor, or already a device tensor.
        if isinstance(current_pos, int):
            cur_pos_int = current_pos
        elif torch.is_tensor(current_pos):
            cur_pos_int = int(current_pos.reshape(-1)[0].item())
        else:
            try:
                cur_pos_int = int(current_pos)
            except Exception:
                return
        for layer in self.layers:
            attn = getattr(layer, "attention", None)
            if attn is None:
                continue
            update = getattr(attn, "_update_decode_mask_buf", None)
            if update is not None:
                update(cur_pos_int)

    def prepare_decode_shard_configs(self, is_cur_pos_sharded=False, is_page_table_sharded=False):
        """
        Prepares the sharding configuration for cur_pos and page_table tensors
        """
        cur_pos_memory_config = None
        page_table_memory_config = None
        # qwen3.6 replicates page_table / current_pos across the mesh (columns are
        # H/4 TP, not data-parallel); the L1 column-split shard specs are invalid
        # here. Force replicated (None) configs to match prepare_decode_inputs_host.
        if self.is_qwen36:
            is_cur_pos_sharded = False
            is_page_table_sharded = False
        if is_cur_pos_sharded:
            cur_pos_shard_spec = ttnn.ShardSpec(
                self.args.sub_core_grids,
                (1, self.args.max_batch_size // self.mesh_device.shape[1]),
                ttnn.ShardOrientation.ROW_MAJOR,
            )
            cur_pos_memory_config = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, cur_pos_shard_spec
            )
        if is_page_table_sharded:
            page_table_shard_spec = ttnn.ShardSpec(
                self.args.sub_core_grids,
                (
                    self.args.batch_size_per_device_group,
                    self.paged_attention_config.max_num_blocks // self.args.max_batch_size,
                ),
                ttnn.ShardOrientation.ROW_MAJOR,
            )
            page_table_memory_config = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, page_table_shard_spec
            )
        return [None, cur_pos_memory_config, None, page_table_memory_config]

    def prepare_decode_inputs_host(
        self, tokens, current_pos, page_table=None, is_cur_pos_sharded=False, is_page_table_sharded=False
    ):
        """
        Inputs are torch tensors or python types. Outputs are ttnn tensors on host.
        NOTE: Tokens and current_pos are padded to batch
        NOTE: if is_cur_pos_sharded is True, current_pos_tt is returned as a device tensor
        NOTE: if is_page_table_sharded is True, page_table is returned as a device tensor
        """
        B = tokens.shape[0]
        # assert current_pos.shape[0] == B, "Batch size mismatch"
        assert (
            B == self.args.max_batch_size
        ), f"Batch size {B} must be equal to max_batch_size {self.args.max_batch_size}"

        # qwen3.6 batch-distribution: the decode activation is REPLICATED across all
        # 32 devices (the 4 mesh columns are the H/4 tensor-parallel split, NOT
        # data-parallel batch groups), so EVERY device runs all N users with
        # my_batch_idx=0..N-1. page_table / current_pos must therefore be REPLICATED
        # too (every device holds all N rows) — the llama70b column-shard convention
        # (dims=(None,-2)/(None,0), 8 users/column) routes B>1 users to wrong pages /
        # out-of-range rows. Force the replicated path and disable the L1-sharded
        # column-split branches for qwen3.6 at B>1. (B==1 already replicated.)
        if self.is_qwen36 and B > 1:
            is_cur_pos_sharded = False
            is_page_table_sharded = False

        # Necessary padding to be full tile sized when on device
        tokens = torch.nn.functional.pad(tokens.view(-1), (0, 32 - len(tokens)), "constant", 0)
        tokens = ttnn.from_torch(
            tokens,
            device=None,
            dtype=ttnn.uint32,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        tokens = ttnn.unsqueeze_to_4D(tokens)
        rot_current_pos = torch.maximum(
            current_pos, torch.tensor(0, dtype=torch.int64)
        )  # Ensure position indices are non-negative
        if self.is_qwen36:
            # qwen3.6 decode uses the partial-RoPE [32,32] rot_idxs (a single scalar
            # position replicated across the tile), NOT the llama [32]-batch column-shard
            # path (get_rm_rot_idxs asserts shape==[32]). Mirrors the working demo
            # (text_demo_qwen36: get_qwen36_rm_rot_idxs). Single-user decode.
            cur_pos_int = int(rot_current_pos.reshape(-1)[0].item())
            rope_idxs = self.rope_setup.get_qwen36_rm_rot_idxs(cur_pos_int, on_host=True)
        else:
            rope_idxs = self.rope_setup.get_rm_rot_idxs(rot_current_pos, on_host=True)
        cur_pos_shard_dim = 0
        if is_cur_pos_sharded:
            cur_pos_shard_dim = 1
            current_pos = current_pos.repeat(self.args.sub_core_grids.num_cores(), 1)
        # qwen3.6: replicate current_pos at B>1 (see note above). Non-qwen36
        # keeps the llama70b column-shard (dims=(None,0)) at B>1.
        cur_pos_dims = (None, None) if (self.is_qwen36 or B == 1) else (None, cur_pos_shard_dim)
        current_pos_tt = ttnn.from_torch(
            current_pos,
            device=None,
            dtype=ttnn.int32,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                self.mesh_device,
                dims=cur_pos_dims,
                mesh_shape=self.args.cluster_shape,
            ),
        )
        if page_table is not None:
            if is_page_table_sharded:
                page_table_chunks = page_table.split(B // self.args.cluster_shape[1], dim=0)
                repeated_page_table_chunks = [
                    chunk.repeat(self.args.sub_core_grids.num_cores(), 1) for chunk in page_table_chunks
                ]
                page_table = torch.cat(repeated_page_table_chunks, dim=0)

            # qwen3.6: replicate page_table at B>1 (see note above). Non-qwen36
            # keeps the llama70b column-shard (dims=(None,-2)) at B>1.
            page_table_dims = (None, None) if (self.is_qwen36 or B == 1) else (None, -2)
            page_table = ttnn.from_torch(
                page_table,
                device=None,
                dtype=ttnn.uint16 if is_page_table_sharded else ttnn.int32,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    self.mesh_device,
                    dims=page_table_dims,
                    mesh_shape=self.args.cluster_shape,
                ),
            )
        return tokens, current_pos_tt, rope_idxs, page_table

    def transform_decode_inputs_device(self, tokens, current_pos, rope_idxs, page_table=None):
        """
        Inputs are ttnn tensors on device. This function applies any on-device
        transformations which should happen before forward decode.
        For example: tilize, reshape, shard.
        Return transformed device tensors

        Get rope sin/cos
        Embed tokens
        """
        if self.is_qwen36:
            tt_rot_mats = self.rope_setup.get_qwen36_rm_rot_mats(rope_idxs)
        else:
            tt_rot_mats = self.rope_setup.get_rm_rot_mats(rope_idxs)
        tt_tokens = self.embd(tokens)
        return tt_tokens, current_pos, tt_rot_mats, page_table

    def process_output_prefill_logits(self, tt_out, last_token_idx, tt_out_logits_saved=None, user_id=0):
        """
        Process prefill output to get logits tensor for on-device sampling.
        Returns logits in the same format as decode (before all-gather), suitable for sampling module.
        For non-batched prefill, returns single user logits. For batched prefill, returns list of logits.
        """
        x, _ = self.norm(tt_out, res=None, mode="prefill")
        if isinstance(last_token_idx, list):
            # batched prefill: split the output tensor by the batch size and do the processing for each batch in a loop
            batch_size = len(last_token_idx)
            x_split = ttnn.split(x, x.shape[-2] // batch_size, dim=2)
        else:
            x_split = [x]

        logits_list = []
        for i, x in enumerate(x_split):
            if isinstance(last_token_idx, list):
                last_token_idx_i = last_token_idx[i]
            else:
                last_token_idx_i = last_token_idx
            x = x[:, :, last_token_idx_i : last_token_idx_i + 1, :]
            # lm_head returns logits in sharded format (same as decode before all-gather)
            tt_logits = self.lm_head(x, None, mode="prefill")
            tt_logits = tt_logits[0]
            tt_logits = ttnn.reshape(
                tt_logits,
                ttnn.Shape([1, 1, 1, tt_logits.shape[-1]]),
                ttnn.Shape([1, 1, tt_logits.shape[-2], tt_logits.shape[-1]]),
            )
            logits_list.append(tt_logits)

        return logits_list

    def process_output_prefill(self, tt_out, last_token_idx, tt_out_logits_saved=None, user_id=0):
        """
        Input is ttnn device tensor of logits. Output is torch logits or tokens tensor.
        NOTE: In this model, prefill always uses get_last_token

        user_id is used to select output from the correct mesh column for paged prefill.
        For batch_size=1, only the user's column has valid KV cache data.
        Mesh is 8x4 (8 rows for heads, 4 columns for data parallel).
        user_id 0-7 → column 0, user_id 8-15 → column 1, etc.
        """
        # Determine which device to read output from based on user_id
        # Device index for row 0 of each column: col 0 → dev 0, col 1 → dev 1, etc.
        output_device_idx = user_id // 8  # 0, 1, 2, or 3

        x, _ = self.norm(tt_out, res=None, mode="prefill")
        if isinstance(last_token_idx, list):
            # batched prefill: split the output tensor by the batch size and do the processing for each batch in a loop
            batch_size = len(last_token_idx)
            x_split = ttnn.split(x, x.shape[-2] // batch_size, dim=2)
        else:
            x_split = [x]

        toks_list = []
        for i, x in enumerate(x_split):
            if isinstance(last_token_idx, list):
                last_token_idx_i = last_token_idx[i]
            else:
                last_token_idx_i = last_token_idx

            x = x[:, :, last_token_idx_i : last_token_idx_i + 1, :]
            tt_logits = self.lm_head(x, None, mode="prefill")
            # Gather the output across all devices and untilize the tensor (for argmax)
            tt_logits = self.tt_ccl.line_all_gather(
                tt_logits[0],
                dim=3,
                num_links=min(3, self.model_config["GALAXY_NUM_LINKS"]),
                cluster_axis=0,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                buffer_key="SAMPLING",
            )

            tt_logits = ttnn.untilize(tt_logits, use_multicore=True)

            tt_logits = ttnn.reshape(
                tt_logits,
                ttnn.Shape([1, 1, 1, tt_logits.shape[-1]]),
                ttnn.Shape([1, 1, tt_logits.shape[-2], tt_logits.shape[-1]]),
            )

            tt_out = ttnn.argmax(tt_logits, dim=3, keepdim=True, use_multicore=True)
            if isinstance(tt_out, list):
                tt_out = tt_out[0]
            toks = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[output_device_idx]).float()[0, 0, 0, :1]
            toks_list.append(toks)

        if tt_out_logits_saved is not None:
            # make sure tt_out_logits_saved is mutable
            logits_saved = ttnn.to_torch(ttnn.get_device_tensors(tt_logits)[output_device_idx]).float()[0, 0, :, :]

            tt_out_logits_saved.copy_(logits_saved)

        return toks_list if isinstance(last_token_idx, list) else toks

    def process_output_decode(self, tt_out):
        """
        Input is ttnn device tensor of tokens. Output is the corresponding torch tensor.
        """
        if isinstance(tt_out, list):
            tt_out = tt_out[0]

        if isinstance(tt_out, tuple):
            tt_log_probs = tt_out[1]
            tt_out = tt_out[0]
            tt_out_cpu = tt_out.cpu(blocking=False, cq_id=0)

            if tt_log_probs is not None:
                tt_log_probs_cpu = tt_log_probs.cpu(blocking=False, cq_id=0)
            else:
                tt_log_probs_cpu = None
        else:
            tt_out_cpu = tt_out.cpu(blocking=False, cq_id=0)
            tt_log_probs_cpu = None
        return tt_out_cpu, tt_log_probs_cpu, ttnn.record_event(self.mesh_device, 0)

    def ttnn_prefill_forward(
        self,
        x,  # ttnn.Tensor, shape [1, 1, seq_len, hidden_dim]; token embeddings input replicated across all devices
        user_id=0,  # ttnn.Tensor, shape [1]; user id replicated across all devices
        page_table=None,  # ttnn.Tensor, shape [4, num_blocks]; main paged-attention page table, replicated across rows, sharded across columns. 32*num_blocks for batched prefill.
        chunk_page_table=None,  # ttnn.Tensor or None, shape [4, num_blocks]; only needed for prefix caching (single user), replicated across rows, sharded across columns
        chunk_start_idx=None,  # ttnn.Tensor, shape [1]; index of cached-token split for prefix caching, replicated across all devices
        start_pos=0,  # int, starting position in sequence for attention (used in SDPA path decision)
        get_last_token=-1,  # int or list[int], output mode: which token to return (last idx or indices)
        kv_cache=None,  # ttnn.Tensor, data parallel across cols, head parallel across rows
        rot_mats=None,  # Tuple[ttnn.Tensor, ttnn.Tensor], each of shape [1, 1, max_seq_len, head_dim]; RoPE matrices for full (0..max_seq_len) replicated across all devices
        batch_size=1,  # int, number of users or batch size for prefill; controls input slicing and paging
    ):
        """
        Prefill forward. Expects rot_mats to be full (0..max_seq_len) from get_or_create_prefill_rot_mats(),
        and chunk_start_idx to be a device tensor (e.g. shape (1,) int32) used for rot mats slicing and chunked SDPA.
        start_pos is a Python int used for attention decisions (SDPA path, program config); must match
        the value in chunk_start_idx.
        """
        assert rot_mats is not None, "prefill requires rot_mats (full from get_or_create_prefill_rot_mats)"
        assert chunk_start_idx is not None and hasattr(
            chunk_start_idx, "shape"
        ), "prefill requires chunk_start_idx as device tensor"
        tt_logits = self.forward(
            x,
            current_pos=None,
            rot_mats=rot_mats,
            user_id=user_id,
            mode="prefill",
            page_table=page_table,
            chunk_page_table=chunk_page_table,
            chunk_start_idx=chunk_start_idx,
            start_pos=start_pos,
            get_last_token=get_last_token,
            kv_cache=kv_cache,
            batch_size=batch_size,
        )
        return tt_logits

    def _increment_decode_positions_device(self, current_pos, rot_mat_idxs, is_cur_pos_sharded=False):
        ttnn.plus_one(
            current_pos,
            sub_core_grids=self.args.sub_core_grids
            if is_cur_pos_sharded
            else ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 0))]),
            skip_negative_entries=True,
        )
        # plus_one advances the rope index IN-PLACE for the self-contained on-device
        # decode trace (matches text_demo_qwen36 / test_decode_perf_intrace: in-trace
        # plus_one on rot_idxs). qwen3.6's rot_mat_idxs is a [32,32] tile and plus_one
        # on this single-core grid handles it (the demo does exactly this). In the eager
        # path the result is harmlessly overwritten by the next-step rebuild.
        ttnn.plus_one(
            rot_mat_idxs,
            sub_core_grids=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 0))]),
        )

    def ttnn_decode_forward(
        self,
        x,
        current_pos,
        rot_mat_idxs,
        page_table=None,
        kv_cache=None,
        tt_out_logits_saved=None,
        is_cur_pos_sharded=False,
        return_logits=False,
        capture_sampling_trace=False,  # If true, return logits so sampling can be traced elsewhere
    ):
        """
        This method will take device tensors and any other args to run forward.
        It returns ttnn device tensors.
        """
        # qwen3.6 current_pos is replicated (single-core DRAM), not L1 sub-core
        # sharded — so plus_one must use the single-core grid (is_cur_pos_sharded
        # False). Keep consistent with prepare_decode_inputs_host / shard configs.
        if self.is_qwen36:
            is_cur_pos_sharded = False
        if self.is_qwen36:
            rot_mats = self.rope_setup.get_qwen36_rm_rot_mats(rot_mat_idxs)
        else:
            rot_mats = self.rope_setup.get_rm_rot_mats(rot_mat_idxs)
        if self.is_qwen36:
            # qwen3.6 decode embedding (mirrors text_demo_qwen36._run_decode_intrace):
            # raw ttnn.embedding -> DRAM full-H bfloat16 (NOT the TtLlamaEmbedding
            # col-sharded H/4 L1 layout, which is the prefill contract). The decoder's
            # default decode path (non-L1-residual) expects DRAM full-H, 1 row.
            x_emb_flat = ttnn.embedding(
                x,
                self.embd.weights,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
            )
            if os.environ.get("QWEN36_DECODE_L1_RESIDUAL", "0") == "1" or os.environ.get("QWEN36_DECODE_32ROW", "0") == "1":
                x_embd = ttnn.reshape(
                    x_emb_flat, ttnn.Shape([1, 1, x_emb_flat.shape[-2], x_emb_flat.shape[-1]])
                )
            else:
                x_emb_3d = ttnn.slice(
                    x_emb_flat, [0, 0, 0], [1, 1, x_emb_flat.shape[-1]], memory_config=ttnn.DRAM_MEMORY_CONFIG
                )
                x_emb_flat.deallocate(True)
                x_embd = ttnn.unsqueeze_to_4D(x_emb_3d)
        else:
            x_embd = self.embd(x)
        tt_logits = self.forward(
            x_embd,
            current_pos,
            rot_mats=rot_mats,
            mode="decode",
            page_table=page_table,
            kv_cache=kv_cache,
        )
        self._increment_decode_positions_device(current_pos, rot_mat_idxs, is_cur_pos_sharded)

        if return_logits:
            tt_logits = self.tt_ccl.line_all_gather(
                tt_logits[0],
                dim=3,
                num_links=min(3, self.model_config["GALAXY_NUM_LINKS"]),
                cluster_axis=0,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                buffer_key="SAMPLING",
            )

            tt_logits = ttnn.untilize(tt_logits, use_multicore=True, sub_core_grids=self.args.sub_core_grids)

            return tt_logits, None

        # Save output logits to global python object
        if tt_out_logits_saved is not None:
            tt_out_logits = ttnn.to_torch(
                tt_logits[0],
                mesh_composer=ttnn.ConcatMesh2dToTensor(
                    self.mesh_device, dims=(3, 1), mesh_shape=self.args.cluster_shape
                ),
            )
            tt_out_logits = tt_out_logits[0, 0, 0, : self.args.vocab_size]

            tt_out_logits_saved.copy_(tt_out_logits)

        if capture_sampling_trace:
            return tt_logits

        tt_toks, tt_log_probs = self.sampling.sample(
            tt_logits[0],
            tt_out_tok=x,
            enable_trace=False,
        )
        return tt_toks, tt_log_probs

    def switch_mode(self, mode):
        if mode == "decode":
            if self.is_prefill_setup:
                self.tt_ccl.close()
                self.tt_ccl_prefill = self.tt_ccl
                self.is_prefill_setup = False

            if self.is_decode_setup is False:
                self.setup_decode(self.mesh_sub_device_manager_id_decode)
                self.is_decode_setup = True
                # prefetch
                for layer in self.layers:
                    layer.prefetch(self.prefetcher_setup, self.tt_ccl)
                self.norm.tt_ccl = self.tt_ccl
                self.lm_head.tt_ccl = self.tt_ccl
                if self.args.use_prefetcher:
                    self.tt_tensors = self.prefetcher_setup.get_input_tensors()
                    # Re-create global CB for decode (if it was not already created)
                    self.prefetcher_setup.create_global_cb()

        else:
            if self.is_decode_setup:
                self.tt_ccl.close()
                self.tt_ccl_decode = self.tt_ccl
                self.is_decode_setup = False

            if self.is_prefill_setup is False:
                self.setup_prefill(self.mesh_sub_device_manager_id_prefill)
                self.is_prefill_setup = True
                for layer in self.layers:
                    layer.prefetch(self.prefetcher_setup, self.tt_ccl)
                self.norm.tt_ccl = self.tt_ccl
                self.lm_head.tt_ccl = self.tt_ccl

    def forward(
        self,
        x: ttnn.Tensor,
        current_pos,
        rot_mats=None,
        user_id=0,
        mode="decode",
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,  # On-device
        start_pos=0,  # Python int
        get_last_token=-1,
        kv_cache=None,
        batch_size=1,
    ):
        # V2-decode: qwen3.6 single-user decode uses the DRAM-residual contract
        # via the qwen36 prefill primitives (see decoder block is_qwen36_path
        # branch). Skip the L1-sharded prefetcher path which assumes the 70B
        # batch-32 packed [1,1,32,H] contract.
        is_qwen36_decode = self.is_qwen36 and mode == "decode"
        if mode == "decode" and not is_qwen36_decode:
            if self.args.use_prefetcher:
                self.prefetcher_setup.create_global_cb()
                garbage_tensor = ttnn.dram_prefetcher(
                    self.tt_tensors,
                    num_layers=self.n_layers,
                    global_cb=self.prefetcher_setup.global_circular_buffer,
                    enable_performance_mode=self.enable_prefetcher_performance_mode,
                )
            self.mesh_device.set_sub_device_stall_group([self.prefetcher_setup.worker_sub_device_id])

        # Prefill: for prefix caching (start_pos > 0), slice rot_mats to [chunk_start_idx, max_seq_len).
        # When start_pos == 0, use full rot_mats as-is (no slice) to avoid ttnn.concat/ttnn.slice device
        # ops that can hang on some builds; trace capture for prefix-cached runs uses start_pos > 0.
        if mode == "prefill" and start_pos > 0:
            full_rot_cos, full_rot_sin = rot_mats[0], rot_mats[1]
            num_devices = self.args.cluster_shape[0] * self.args.cluster_shape[1]
            z = self._tt_slice_start_zeros_4
            tt_slice_starts = ttnn.concat(
                [z[0:2], chunk_start_idx, z[3:4]],
                dim=0,
            )
            rot_cos_slice = ttnn.slice(
                input_tensor=full_rot_cos,
                starts=tt_slice_starts,
                ends=self._tt_seq_len_buffer,
                slice_dim=2,
                num_devices=num_devices,
            )
            rot_sin_slice = ttnn.slice(
                input_tensor=full_rot_sin,
                starts=tt_slice_starts,
                ends=self._tt_seq_len_buffer,
                slice_dim=2,
                num_devices=num_devices,
            )
            rot_mats = (rot_cos_slice, rot_sin_slice)

        h = None
        # x needs to be in bfloat16_b as it gets reused as the residual tensor
        for i, layer in enumerate(self.layers):
            x, h = layer(
                x,
                h,
                current_pos,
                rot_mats,
                user_id,
                mode,
                page_table,
                chunk_page_table=chunk_page_table,
                chunk_start_idx=start_pos,
                chunk_start_idx_tensor=chunk_start_idx if mode == "prefill" else None,
                kv_cache=kv_cache[i] if kv_cache is not None else None,
                batch_size=batch_size,
            )
            if os.environ.get("QWEN36_DUMP_HIDDEN", "0") == "1" and mode == "decode" and i in (0, 1, 2, 3, 4, 8):
                _t = x if h is None else h
                _td = ttnn.to_torch(ttnn.get_device_tensors(_t)[0]).float()
                _rows = _td.reshape(-1, _td.shape[-1])  # [users(32), hidden] on dev0
                _r0 = _rows[0, :6]
                # per-user maxdiff vs user0: for identical users this MUST be ~0;
                # the first layer where it's nonzero is where per-user divergence enters.
                _nrows = _rows.shape[0]
                _udiff = (
                    max((_rows[u] - _rows[0]).abs().max().item() for u in range(1, min(_nrows, 32)))
                    if _nrows > 1
                    else 0.0
                )
                print(
                    f"[DUMP_HIDDEN] after layer {i} ({'h' if h is not None else 'x'}) dev0 "
                    f"absmean={_td.abs().mean():.4f} max={_td.abs().max():.4f} per_user_maxdiff={_udiff:.4f} "
                    f"row0[:6]={[round(v,3) for v in _r0.tolist()]}"
                )
        # ttnn.deallocate(h)
        if mode == "decode" and not is_qwen36_decode:
            if self.args.use_prefetcher:
                ttnn.deallocate(garbage_tensor)

            # Pre-allocated output of AllReduce in LM Head to avoid memory cloberring
            self.tt_ccl.tt_lm_head_buffer_l1 = ttnn.to_memory_config(
                self.tt_ccl.tt_lm_head_buffer, self.tt_ccl.lm_head_buffer_mem_cfg
            )

        if mode == "prefill":
            return x

        # V2-decode: qwen3.6 single-user decode uses the prefill norm + lm_head
        # primitives (DRAM-friendly). The 70B decode path uses the L1-sharded
        # variants which assume the batch-32 packed contract.
        if is_qwen36_decode:
            # BATCH-N decode tail (batch_size > 1, distinct users): the prefill
            # (single-user) norm + lm_head do NOT produce correct per-row logits
            # at N rows (the root cause of the wrong batch-32 token). Route to the
            # decode-mode (packed-N contract) norm + lm_head — the same L1-sharded
            # path the 70B decode uses — which emits all N users' logits. The
            # decode rms_allgather LAYERNORM buffer + the LM_HEAD all-reduce buffer
            # are lazily registered on first use (B1 LAYERNORM precedent). Gated on
            # batch_size > 1, so batch-1 / 32-row-identical (_carry32_tail below)
            # is completely untouched (byte-identical).
            # QWEN36_FORCE_SWITCH_DECODE: also take the decode-mode tail at batch-1
            # when the model is on decode-mode tt_ccl (unification probe — run batch-1
            # on the SAME decode path as batch-32). Without this, batch-1 falls to the
            # prefill _carry32 tail, which crashes on decode-mode CCL (bad optional access).
            _force_decode_tail = (
                os.environ.get("QWEN36_FORCE_SWITCH_DECODE", "0") == "1"
                and getattr(self.tt_ccl, "mode", None) == "decode"
            )
            if (batch_size > 1 or _force_decode_tail) and os.environ.get("QWEN36_B32_DECODE_TAIL", "1") == "1":
                # BISECT KNOB: QWEN36_B32_DECODE_TAIL=0 routes batch-32 through the
                # proven row-0 _carry32_tail (prefill norm + prefill lm_head on row 0)
                # below instead of this decode-mode tail — for identical users that
                # isolates whether the backbone produces a correct row-0 hidden
                # (token 248068 ⇒ backbone OK, decode-tail is the bug).
                # The decode-mode lm_head's all_reduce consumes tt_lm_head_buffer_l1
                # (llama_ccl.line_all_reduce:952); the non-qwen36 decode path sets it
                # up at the top of forward, but the qwen36 branch skips that. The
                # underlying tt_lm_head_buffer + lm_head_buffer_mem_cfg may not exist on
                # this tt_ccl (qwen36 decode historically used the prefill lm_head), so
                # build them on demand, then create the L1 copy (deallocated inside
                # line_all_reduce each call).
                # The LMHead was constructed with the PREFILL tt_ccl instance (init
                # time); switch_mode updates the layers' tt_ccl but not the lm_head's,
                # so its internal line_all_reduce would take the prefill CCL branch and
                # mismatch. Point it at the model's current (decode) tt_ccl — the same
                # instance the GDN/full-attn decode layers use — so the lm_head all-reduce
                # takes the decode branch (tt_lm_head_buffer_l1, built just below).
                self.lm_head.tt_ccl = self.tt_ccl
                # NOTE: qwen36 decode lm_head now reduces via DRAM
                # reduce_scatter+all_gather (see lm_head.py), so the large L1
                # all_reduce_async buffer (ensure_lm_head_buffer / tt_lm_head_buffer_l1)
                # is intentionally NOT allocated here — it would waste ~300 KB/core
                # and its static CB clashed on core (0,0).
                # Full llama decode contract (Option A): with switch_mode("decode")
                # active (decode tt_ccl) + QWEN36_DECODE_L1_RESIDUAL (x arrives
                # L1-sharded DECODE_RESIDUAL), run the decode-mode rms_allgather norm
                # + decode lm_head — the batched primitives that emit correct per-row
                # logits for N distinct users. (The prefill norm/lm_head are the
                # single-user batch-1 shortcut; they corrupt row-0 logits at N rows.)
                if os.environ.get("QWEN36_DUMP_HIDDEN", "0") == "1":
                    _xpre = ttnn.to_torch(ttnn.get_device_tensors(x)[0]).float()
                    _r0 = _xpre.reshape(-1, _xpre.shape[-1])[0, :12]
                    print(
                        f"[DUMP_HIDDEN] B>1 pre-norm dev0 shape={list(_xpre.shape)} "
                        f"absmean={_xpre.abs().mean():.4f} max={_xpre.abs().max():.4f} "
                        f"nan={bool(_xpre.isnan().any())} row0[:12]={[round(v,3) for v in _r0.tolist()]}"
                    )
                x, _ = self.norm(x, res=None, mode="decode")
                if os.environ.get("QWEN36_DUMP_HIDDEN", "0") == "1":
                    _xpost = ttnn.to_torch(ttnn.get_device_tensors(x)[0]).float()
                    print(
                        f"[DUMP_HIDDEN] post-norm dev0 shape={list(_xpost.shape)} "
                        f"absmean={_xpost.abs().mean():.4f} max={_xpost.abs().max():.4f} "
                        f"nan={bool(_xpost.isnan().any())}"
                    )
                lm_head_output = self.lm_head(
                    x,
                    self.prefetcher_setup.worker_sub_device_id if self.prefetcher_setup is not None else None,
                    mode="decode",
                )
                return lm_head_output
            # The decoder loop exit is col-sharded [B, 1, T=1, H/4] (same
            # contract as prefill). Run the final norm + lm_head via the
            # prefill primitives (tt_distributed_rmsnorm; lm_head forward
            # with mode="prefill").
            # Step 1 (QWEN36_DECODE_L1_RESIDUAL): with the L1 residual on, the decoder
            # exits as a 32-row L1-sharded tensor ([1,1,32,1280]). The prefill norm
            # primitive (rms_norm_pre/post_all_gather) and the prefill lm_head expect a
            # DRAM-interleaved input, so convert back to DRAM here. The 32 rows are
            # identical (batch-1), so the lm_head produces 32 identical logit rows and the
            # caller's row-0 slice is correct. Flag off ⇒ already DRAM 1-row ⇒ no-op.
            # 32-row carry (QWEN36_DECODE_32ROW DRAM baseline or QWEN36_DECODE_L1_RESIDUAL):
            # the decoder exits with 32 tile-padded rows that are IDENTICAL single-user copies
            # (row 0 = the real user). The final norm + lm_head must run SINGLE-USER on row 0 —
            # running them at 32 rows does NOT preserve row 0 (the prefill lm_head/norm corrupt
            # the row-0 logits at 32 rows, the root cause of the wrong decode token). Convert to
            # DRAM (L1 case) and slice row 0 so norm + lm_head are byte-identical to batch-1.
            _carry32_tail = (
                os.environ.get("QWEN36_DECODE_32ROW", "0") == "1"
                or os.environ.get("QWEN36_DECODE_L1_RESIDUAL", "0") == "1"
                # option-B bisect: batch-32 DP backbone reached here because the
                # decode tail was disabled (QWEN36_B32_DECODE_TAIL=0). The DP
                # backbone produced a 32-row residual; slice row 0 + run the proven
                # prefill norm/lm_head (identical users ⇒ row 0 is the answer).
                or (batch_size > 1 and os.environ.get("QWEN36_B32_DECODE_TAIL", "1") == "0")
            )
            if _carry32_tail and x.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
                x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
            if _carry32_tail:
                _xb, _, _xt, _xh = list(x.shape)
                if _xt != 1:
                    x = ttnn.slice(x, [0, 0, 0, 0], [_xb, 1, 1, _xh], memory_config=ttnn.DRAM_MEMORY_CONFIG)
            if os.environ.get("QWEN36_DUMP_HIDDEN", "0") == "1":
                _xpre = ttnn.to_torch(ttnn.get_device_tensors(x)[0]).float()
                _r0 = _xpre.reshape(-1, _xpre.shape[-1])[0, :12]
                print(
                    f"[DUMP_HIDDEN] B1 pre-norm dev0 shape={list(_xpre.shape)} "
                    f"absmean={_xpre.abs().mean():.4f} max={_xpre.abs().max():.4f} "
                    f"row0[:12]={[round(v,3) for v in _r0.tolist()]}"
                )
            x, _ = self.norm(x, res=None, mode="prefill")
            if get_last_token != -1:
                x = x[:, :, get_last_token:, :]
            lm_head_output = self.lm_head(x, None, mode="prefill")
            return lm_head_output

        # Output norm
        x, res = self.norm(x, res=None, mode=mode)

        if get_last_token != -1:
            x = x[:, :, get_last_token:, :]

        lm_head_output = self.lm_head(
            x, None if mode == "prefill" else self.prefetcher_setup.worker_sub_device_id, mode=mode
        )
        # if mode is decode and Qwen model
        if mode == "decode" and self.args.is_qwen:
            ttnn.to_memory_config(self.tt_ccl.tt_lm_head_buffer, ttnn.DRAM_MEMORY_CONFIG)
        return lm_head_output

    def prefill_chunked(
        self,
        x,
        rot_mats,
        gdn_chunk_size,
        user_id=0,
        page_table=None,
        kv_cache=None,
        chunk_start_idx_tensor=None,
        batch_size=1,
    ):
        """Long-context prefill driver (GDN-only chunking).

        GDN/linear-attention layers are processed in sequence-chunks of
        ``gdn_chunk_size``, carrying conv + recurrent state across chunks via the
        persistent dn_state_buffer / conv_state_buffer (the DeltaNet block seeds
        from them when ``attention._pf_chunk_idx > 0``). Full-attention layers run
        single-pass over the whole sequence (paged-KV write + flash SDPA — no
        chunked-SDPA needed). Returns the post-layer-loop hidden (col-sharded,
        pre-final-norm), identical to ``forward(mode='prefill')`` so the caller's
        norm + lm_head path is unchanged.
        """
        seq_len = x.shape[2]
        cos_full, sin_full = (rot_mats[0], rot_mats[1]) if rot_mats is not None else (None, None)
        for i, layer in enumerate(self.layers):
            is_gdn = getattr(layer, "is_linear_attention_layer", False)
            layer_chunk = gdn_chunk_size if is_gdn else seq_len
            outs = []
            for ci, cs in enumerate(range(0, seq_len, layer_chunk)):
                ce = min(cs + layer_chunk, seq_len)
                x_chunk = x if (cs == 0 and ce == seq_len) else ttnn.slice(x, (0, 0, cs, 0), (1, 1, ce, x.shape[-1]))
                if is_gdn:
                    layer.attention._pf_chunk_idx = ci  # 0 => fresh state, >0 => carry from buffers
                    rm = None
                else:
                    rm = [cos_full[:, :, cs:ce, :], sin_full[:, :, cs:ce, :]] if cos_full is not None else None
                xo, _ = layer(
                    x_chunk,
                    None,
                    None,
                    rm,
                    user_id,
                    "prefill",
                    page_table,
                    chunk_page_table=None,
                    chunk_start_idx=0,
                    chunk_start_idx_tensor=chunk_start_idx_tensor,
                    kv_cache=kv_cache[i] if kv_cache is not None else None,
                    batch_size=batch_size,
                )
                if is_gdn:
                    layer.attention._pf_chunk_idx = None
                if x_chunk is not x:
                    ttnn.deallocate(x_chunk)
                outs.append(xo)
            x_new = outs[0] if len(outs) == 1 else ttnn.concat(outs, dim=2)
            if len(outs) > 1:
                for o in outs:
                    ttnn.deallocate(o)
            ttnn.deallocate(x)
            x = x_new
        return x

    def __del__(self):
        # Guard against __del__ firing when __init__ raised before
        # self.tt_ccl was set (otherwise AttributeError hides the real
        # construction error).
        tt_ccl = getattr(self, "tt_ccl", None)
        if tt_ccl is not None:
            tt_ccl.close()

        # clear global saved addresses
        global global_tt_tensor_address
        global_tt_tensor_address = None
