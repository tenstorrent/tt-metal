## Migrate ops to a single latest infra (`tt-metal` #32680)

- **State**: open
- **Assignee**: @ayerofieiev-tt
- **Link**: https://github.com/tenstorrent/tt-metal/issues/32680

### Short description

- **What**: Consolidate how Device Operations are defined by migrating from the old Device Operation structure (vector-based inputs/outputs, extra heap allocations) to the newer TMP-style Device Operation with explicit `operation_attributes_t`, `tensor_args_t`, typed return values, and program factories.
- **Why**:
  - It is confusing to maintain two parallel Device Operation structures.
  - It makes large-scale refactors and AI-assisted development harder.
  - The new Device Operation structure enables more precise input/output typing, fewer heap allocations, and cleaner program factory/hash/validation patterns.

For full migration details (examples of `UnaryDeviceOperation`, `operation_attributes_t`, `tensor_args_t`, `program_factory_t`, hash computation, and program factory overrides) see the original issue body.

---
Great example of the OP done as TMP
+ https://github.com/tenstorrent/tt-metal/tree/main/ttnn/cpp/ttnn/operations/experimental/dropout

### Sub-issues (tasks under #32680)

Checkbox reflects current GitHub state (`open` → unchecked, `closed` → checked).

- [ ] **#32683** `transformer/sdpa/device/joint_sdpa_op.hpp`
  - **State**: open
  - **Assignee**: @awliu-TT
  - **Link**: https://github.com/tenstorrent/tt-metal/issues/32683

- [ ] **#32684** `transformer/sdpa/device/ring_joint_sdpa_op.hpp`
  - **State**: open
  - **Assignee**: @awliu-TT
  - **Link**: https://github.com/tenstorrent/tt-metal/issues/32684

- [ ] **#32685** `transformer/sdpa/device/sdpa_op.hpp`
  - **State**: open
  - **Assignee**: @awliu-TT
  - **Link**: https://github.com/tenstorrent/tt-metal/issues/32685

- [ ] **#32686** `transformer/sdpa_decode/device/sdpa_decode_op.hpp`
  - **State**: open
  - **Assignee**: @awliu-TT
  - **Link**: https://github.com/tenstorrent/tt-metal/issues/32686

- [ ] **#32687** `embedding/device/embedding_device_operation.hpp`
  - **State**: open
  - **Assignee**: @awliu-TT
  - **Link**: https://github.com/tenstorrent/tt-metal/issues/32687

- [ ] **#32688** `embedding_backward/device/embedding_backward_device_operation.hpp`
  - **State**: open
  - **Assignee**: @awliu-TT
  - **Link**: https://github.com/tenstorrent/tt-metal/issues/32688

- [ ] **#32689** `kv_cache/device/update_cache_op.hpp`
  - **State**: open
  - **Assignee**: @awliu-TT
  - **Link**: https://github.com/tenstorrent/tt-metal/issues/32689

- [ ] **#32690** `prefetcher/prefetcher/device/dram_prefetcher_op.hpp`
  - **State**: open
  - **Assignee**: @awliu-TT
  - **Link**: https://github.com/tenstorrent/tt-metal/issues/32690

- [ ] **#32719** `experimental/transformer/split_query_key_value_and_split_heads/device/split_query_key_value_and_split_heads_device_operation.hpp`
  - **State**: open
  - **Assignee**: @awliu-TT
  - **Link**: https://github.com/tenstorrent/tt-metal/issues/32719

- [ ] **#32691** `normalization/groupnorm/device/groupnorm_op.hpp`
  - **State**: open
  - **Assignee**: @imichalakTT
  - **Link**: https://github.com/tenstorrent/tt-metal/issues/32691

- [ ] **#32692** `normalization/layernorm/device/layernorm_op.hpp`
  - **State**: open
  - **Assignee**: @imichalakTT
  - **Link**: https://github.com/tenstorrent/tt-metal/issues/32692

- [ ] **#32693** `normalization/layernorm_distributed/device/layernorm_post_all_gather_op.hpp`
  - **State**: open
  - **Assignee**: @imichalakTT
  - **Link**: https://github.com/tenstorrent/tt-metal/issues/32693

- [ ] **#32694** `normalization/layernorm_distributed/device/layernorm_pre_all_gather_op.hpp`
  - **State**: open
  - **Assignee**: @imichalakTT
  - **Link**: https://github.com/tenstorrent/tt-metal/issues/32694

- [ ] **#32695** `normalization/softmax/device/softmax_op.hpp`
  - **State**: open
  - **Assignee**: @imichalakTT
  - **Link**: https://github.com/tenstorrent/tt-metal/issues/32695

- [ ] **#32696** `reduction/argmax/device/argmax_op.hpp`
  - **State**: open
  - **Assignee**: @imichalakTT
  - **Link**: https://github.com/tenstorrent/tt-metal/issues/32696

- [ ] **#32697** `reduction/generic/device/reduce_op.hpp`
  - **State**: open
  - **Assignee**: @imichalakTT
  - **Link**: https://github.com/tenstorrent/tt-metal/issues/32697

- [ ] **#32698** `reduction/moe/device/moe_op.hpp`
  - **State**: open
  - **Assignee**: @imichalakTT
  - **Link**: https://github.com/tenstorrent/tt-metal/issues/32698

- [ ] **#32699** `reduction/prod/device/prod_nc_op.hpp`
  - **State**: open
  - **Assignee**: @imichalakTT
  - **Link**: https://github.com/tenstorrent/tt-metal/issues/32699

- [ ] **#32700** `reduction/prod/device/prod_op_all.hpp`
  - **State**: open
  - **Assignee**: @imichalakTT
  - **Link**: https://github.com/tenstorrent/tt-metal/issues/32700

- [ ] **#32701** `reduction/sampling/device/sampling_op.hpp`
  - **State**: open
  - **Assignee**: @imichalakTT
  - **Link**: https://github.com/tenstorrent/tt-metal/issues/32701

- [ ] **#32702** `reduction/topk/device/topk_op.hpp`
  - **State**: open
  - **Assignee**: @imichalakTT
  - **Link**: https://github.com/tenstorrent/tt-metal/issues/32702

- [ ] **#32703** `experimental/transformer/all_reduce_create_qkv_heads/device/all_reduce_create_qkv_heads_op.hpp`
  - **State**: open
  - **Assignee**: @imichalakTT
  - **Link**: https://github.com/tenstorrent/tt-metal/issues/32703

- [x] **#32704** `experimental/transformer/concatenate_heads/device/concatenate_heads_device_operation.hpp`
  - **State**: closed (completed)
  - **Assignee**: @ssundaramTT
  - **Link**: https://github.com/tenstorrent/tt-metal/issues/32704

- [ ] **#32705** `experimental/transformer/create_qkv_heads/device/create_qkv_heads_device_operation.hpp`
  - **State**: open
  - **Assignee**: @ssundaramTT
  - **Link**: https://github.com/tenstorrent/tt-metal/issues/32705

- [ ] **#32706** `experimental/transformer/create_qkv_heads_from_separate_tensors/device/create_qkv_heads_from_separate_tensors_device_operation.hpp`
  - **State**: open
  - **Assignee**: @ssundaramTT
  - **Link**: https://github.com/tenstorrent/tt-metal/issues/32706

- [ ] **#32707** `experimental/transformer/nlp_concat_heads/device/nlp_concat_heads_device_operation.hpp`
  - **State**: open
  - **Assignee**: @ssundaramTT
  - **Link**: https://github.com/tenstorrent/tt-metal/issues/32707

- [ ] **#32708** `experimental/transformer/nlp_concat_heads_boltz/device/nlp_concat_heads_boltz_device_operation.hpp`
  - **State**: open
  - **Assignee**: @ssundaramTT
  - **Link**: https://github.com/tenstorrent/tt-metal/issues/32708

- [ ] **#32709** `experimental/transformer/nlp_concat_heads_decode/device/nlp_concat_heads_decode_device_operation.hpp`
  - **State**: open
  - **Assignee**: @ssundaramTT
  - **Link**: https://github.com/tenstorrent/tt-metal/issues/32709

- [ ] **#32710** `experimental/transformer/nlp_create_qkv_heads_decode/device/nlp_create_qkv_heads_decode_device_operation.hpp`
  - **State**: open
  - **Assignee**: @ssundaramTT
  - **Link**: https://github.com/tenstorrent/tt-metal/issues/32710

- [ ] **#32711** `experimental/transformer/nlp_create_qkv_heads_falcon7b/device/nlp_create_qkv_heads_falcon7b_device_operation.hpp`
  - **State**: open
  - **Assignee**: @ssundaramTT
  - **Link**: https://github.com/tenstorrent/tt-metal/issues/32711

- [ ] **#32712** `experimental/transformer/nlp_create_qkv_heads_segformer/device/nlp_create_qkv_heads_segformer_device_operation.hpp`
  - **State**: open
  - **Assignee**: @ssundaramTT
  - **Link**: https://github.com/tenstorrent/tt-metal/issues/32712

- [ ] **#32713** `experimental/transformer/nlp_create_qkv_heads_vit/device/nlp_create_qkv_heads_vit_device_operation.hpp`
  - **State**: open
  - **Assignee**: @ssundaramTT
  - **Link**: https://github.com/tenstorrent/tt-metal/issues/32713

- [ ] **#32714** `experimental/transformer/nlp_kv_cache_load_slice/device/nlp_kv_cache_load_slice_device_operation.hpp`
  - **State**: open
  - **Assignee**: @ssundaramTT
  - **Link**: https://github.com/tenstorrent/tt-metal/issues/32714

- [ ] **#32715** `experimental/transformer/rotary_embedding/device/rotary_embedding_device_operation.hpp`
  - **State**: open
  - **Assignee**: @ssundaramTT
  - **Link**: https://github.com/tenstorrent/tt-metal/issues/32715

- [ ] **#32716** `experimental/transformer/rotary_embedding_llama/device/rotary_embedding_llama_device_operation.hpp`
  - **State**: open
  - **Assignee**: @ssundaramTT
  - **Link**: https://github.com/tenstorrent/tt-metal/issues/32716

- [ ] **#32717** `experimental/transformer/rotary_embedding_llama_fused_qk/device/rotary_embedding_llama_fused_qk_device_operation.hpp`
  - **State**: open
  - **Assignee**: @ssundaramTT
  - **Link**: https://github.com/tenstorrent/tt-metal/issues/32717

- [ ] **#32718** `experimental/transformer/rotate_half/device/rotate_half_device_operation.hpp`
  - **State**: open
  - **Assignee**: @ssundaramTT
  - **Link**: https://github.com/tenstorrent/tt-metal/issues/32718

+ [x] **#32720** `experimental/cnn/convert_to_chw/device/convert_to_chw_op.hpp`
  - **State**: closed (completed)
  - **Assignee**: @ayerofieiev-tt
  - **Link**: https://github.com/tenstorrent/tt-metal/issues/32720 https://github.com/tenstorrent/tt-metal/pull/32789

+ [x] **#32721** `experimental/cnn/convert_to_hwc/device/convert_to_hwc_op.hpp`
  - **State**: closed (completed)
  - **Assignee**: @ayerofieiev-tt
  - **Link**: https://github.com/tenstorrent/tt-metal/issues/32721

- [x] **#32722** `experimental/conv3d/device/conv3d_device_operation.hpp`
  - **State**: closed (completed)
  - **Assignee**: @ayerofieiev-tt
  - **Link**: https://github.com/tenstorrent/tt-metal/issues/32722

- [x] **#32723** `experimental/matmul/attn_matmul/device/attn_matmul_device_operation.hpp`
  - **State**: closed (completed)
  - **Assignee**: @ayerofieiev-tt
  - **Link**: https://github.com/tenstorrent/tt-metal/issues/32723

- [x] **#32724** `experimental/matmul/group_attn_matmul/device/group_attn_matmul_device_operation.hpp`
  - **State**: closed (completed)
  - **Assignee**: @ayerofieiev-tt
  - **Link**: https://github.com/tenstorrent/tt-metal/issues/32724

- [x] **#32725** `experimental/padded_slice/device/padded_slice_op.hpp`
  - **State**: closed (completed)
  - **Assignee**: @ayerofieiev-tt
  - **Link**: https://github.com/tenstorrent/tt-metal/issues/32725

- [ ] **#32726** `experimental/paged_cache/device/paged_cache_operation.hpp`
  - **State**: open
  - **Assignee**: @ayerofieiev-tt
  - **Link**: https://github.com/tenstorrent/tt-metal/issues/32726

- [x] **#32727** `experimental/plusone/device/plusone_op.hpp`
  - **State**: closed (completed)
  - **Assignee**: @ayerofieiev-tt
  - **Link**: https://github.com/tenstorrent/tt-metal/issues/32727

- [x] **#32728** `experimental/reduction/fast_reduce_nc/device/fast_reduce_nc_device_operation.hpp`
  - **State**: closed (completed)
  - **Assignee**: @ayerofieiev-tt
  - **Link**: https://github.com/tenstorrent/tt-metal/issues/32728

- [x] **#32729** `experimental/slice_write/device/slice_write_op.hpp`
  - **State**: closed (completed)
  - **Assignee**: @ayerofieiev-tt
  - **Link**: https://github.com/tenstorrent/tt-metal/issues/32729

- [x] **#32730** `experimental/ssm/hc_sum_reduce/device/hc_sum_reduce_op.hpp`
  - **State**: closed (completed)
  - **Assignee**: @ayerofieiev-tt
  - **Link**: https://github.com/tenstorrent/tt-metal/issues/32730

- [x] **#32731** `experimental/ssm/prefix_scan/device/prefix_scan_op.hpp`
  - **State**: closed (completed)
  - **Assignee**: @ayerofieiev-tt
  - **Link**: https://github.com/tenstorrent/tt-metal/issues/32731

- [x] **#32732** `experimental/ssm/repeat_and_interleave_eltwise_mul/device/repeat_and_interleave_eltwise_mul_op.hpp`
  - **State**: closed (completed)
  - **Assignee**: @ayerofieiev-tt
  - **Link**: https://github.com/tenstorrent/tt-metal/issues/32732
