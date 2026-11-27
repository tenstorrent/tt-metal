#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
#
# Run the full set of 4x4 [RING, RING] test rows on a 16-chip sub-torus of one BH Galaxy, by hand.
#
# CI runs only a small part of this set (the GLM components and GLM chunked perf legs, see
# tests/pipeline_reorg/blaze_models_prefill_tests.yaml);
# the rest is kept here so the new rows stay runnable. Each section is a group of test files run through
# run_subtorus_4x4_tests.sh, which narrows the Galaxy to the carve (SUBTORUS_4X4_ENV of the AttnRes harness) and fails
# unless every selected row passed and none skipped.
#
# Usage, from TT_METAL_HOME, on an idle Galaxy:
#   models/demos/deepseek_v3_d_p/tests/run_subtorus_4x4_all.sh --list
#   models/demos/deepseek_v3_d_p/tests/run_subtorus_4x4_all.sh [--keep-going] [section ...]
# With no section, every section runs in the order listed. A section's wall-clock is printed when it ends.
#
# The durations below are the wall-clock of the section on an exabox Galaxy (JIT cache warm after the first section,
# weights from Weka); CI runners are slower, the K3 MoE section by about 1.5x.
set -uo pipefail

HELPER=models/demos/deepseek_v3_d_p/tests/run_subtorus_4x4_tests.sh
T=models/demos/deepseek_v3_d_p/tests
X=tests/ttnn/unit_tests/operations/experimental

# name | expected seconds | what it runs
SECTIONS=(
  "subtorus_ccl_attn_res|593|4x4 RING-RING sub-torus high_bw_all_gather and AttnRes contract, determinism, gather-softmax and block-stats accuracy"
  "subtorus_small_ops|869|4x4 RING-RING sub-torus padded and zero-padded KV cache, masked bincount, offset cumsum and combine op accuracy"
  "subtorus_dispatch_combine|1637|4x4 RING-RING sub-torus prefill dispatch and TTNN dispatch/combine accuracy"
  "subtorus_moe|958|4x4 RING-RING sub-torus TTNN MoE module accuracy"
  "subtorus_modules|840|4x4 RING-RING sub-torus FFN, MoE gate, routing setup, embedding, RMSNorm, shared expert and KV cache table accuracy"
  "subtorus_kda_components|248|4x4 RING-RING sub-torus KDA QKV convolution exchange and recurrence accuracy"
  "subtorus_kda_layers|450|4x4 RING-RING sub-torus KDA layer acceptance, actual start, padding and distributed accuracy"
  "subtorus_glm_sparse_mla|371|4x4 RING-RING sub-torus GLM sparse MLA accuracy"
  "subtorus_glm_block|318|4x4 RING-RING sub-torus GLM prefill block accuracy"
  "subtorus_glm_mtp|563|4x4 RING-RING sub-torus GLM MTP module and multiturn accuracy"
  "subtorus_glm_prefill_transformer|120|4x4 RING-RING sub-torus GLM prefill transformer cache build and accuracy"
  "subtorus_glm_chunked|232|4x4 RING-RING sub-torus GLM chunked prefill transformer accuracy"
  "subtorus_kimi_mla|192|4x4 RING-RING sub-torus Kimi MLA accuracy"
  "subtorus_kimi_dflash|387|4x4 RING-RING sub-torus Kimi DFlash and multiturn accuracy"
  "subtorus_sparse_kv_push_xy4|199|sparse KV push xy4 rows (also run by the GLM components CI leg)"
  # last: perf, needs a 14 kW Galaxy (PREFILL_PERF_IGNORE_POWER=1 for bring-up); also run by the GLM chunked perf CI leg
  "subtorus_glm_moe_perf|190|4x4 RING-RING sub-torus GLM MoE block host and per-op perf"
)

section_subtorus_sparse_kv_push_xy4() {
  "$HELPER" --timeout-s 600 --expect 10 $X/test_sparse_kv_push.py -k "not perf"
}

section_subtorus_ccl_attn_res() {
  export MESH_DEVICE=TG
  "$HELPER" --timeout-s 400 $X/test_high_bw_all_gather.py || return $?
  "$HELPER" --timeout-s 400 $T/attn_res/model/test_attn_res_contract.py || return $?
  "$HELPER" --timeout-s 400 $X/test_attn_res_determinism.py || return $?
  "$HELPER" --timeout-s 400 $X/test_attn_res_gather_softmax.py || return $?
  "$HELPER" --timeout-s 400 $X/test_attn_res_block_stats.py || return $?
}

section_subtorus_small_ops() {
  export MESH_DEVICE=TG
  "$HELPER" --timeout-s 600 --allow-skip "CI runs only the small padded_partial case" $T/op_unit_tests/test_deepseek_prefill_update_padded_kv_cache.py || return $?
  "$HELPER" --timeout-s 600 $T/op_unit_tests/test_zero_padded_kv_cache.py || return $?
  "$HELPER" --timeout-s 300 $T/op_unit_tests/test_masked_bincount.py || return $?
  "$HELPER" --timeout-s 300 --allow-skip "experts do not divide across" $T/op_unit_tests/test_offset_cumsum.py || return $?
  "$HELPER" --timeout-s 900 $T/op_unit_tests/test_combine_fabric2d.py || return $?
}

section_subtorus_dispatch_combine() {
  export MESH_DEVICE=TG
  "$HELPER" --timeout-s 1500 --allow-skip "experts do not divide across" $T/op_unit_tests/test_prefill_dispatch.py || return $?
  "$HELPER" --timeout-s 1800 $T/op_unit_tests/test_ttnn_dispatch_combine.py || return $?
}

section_subtorus_moe() {
  export MESH_DEVICE=TG
  "$HELPER" --timeout-s 2300 $T/pcc/test_ttnn_moe.py || return $?
}

section_subtorus_modules() {
  export MESH_DEVICE=TG
  "$HELPER" --timeout-s 500 $T/pcc/test_ffn.py || return $?
  "$HELPER" --timeout-s 700 $T/pcc/test_moe_gate_prefill2d.py || return $?
  "$HELPER" --timeout-s 400 $T/pcc/test_moe_routing_setup.py || return $?
  "$HELPER" --timeout-s 400 $T/pcc/test_parallel_embedding.py || return $?
  "$HELPER" --timeout-s 300 $T/pcc/test_rmsnorm.py || return $?
  "$HELPER" --timeout-s 300 $T/pcc/test_shared_expert.py || return $?
  "$HELPER" --timeout-s 300 $T/test_kv_cache_table.py || return $?
}

section_subtorus_kda_components() {
  export MESH_DEVICE=TG
  "$HELPER" --timeout-s 600 $T/kda/components/test_qkv_convolution_exchange.py || return $?
  "$HELPER" --timeout-s 400 $T/kda/components/test_recurrence.py || return $?
}

section_subtorus_kda_layers() {
  export MESH_DEVICE=TG
  export KIMI_K3_CKPT=/mnt/weka/model-weights/llm/moonshotai/Kimi-K3-dequantized
  "$HELPER" --timeout-s 700 $T/kda/layer/test_acceptance.py || return $?
  "$HELPER" --timeout-s 500 $T/kda/layer/test_actual_start.py || return $?
  "$HELPER" --timeout-s 300 $T/kda/layer/test_padding_early_exit.py || return $?
  "$HELPER" --timeout-s 300 $T/kimi_k3/test_kda_padding.py || return $?
  "$HELPER" --timeout-s 500 $T/kda/layer/test_distributed.py || return $?
}

section_subtorus_glm_sparse_mla() {
  export MESH_DEVICE=TG
  "$HELPER" --timeout-s 600 $T/sparse_mla/test_sparse_mla.py || return $?
}

section_subtorus_glm_block() {
  export MESH_DEVICE=TG
  export GLM53_HF_MODEL=/mnt/weka/model-weights/llm/zai-org/GLM-5.3-fp8-aca966e4
  export TT_GLM53_PREFILL_TTNN_CACHE=/tmp/glm53_4x4_prefill_cache
  export TT_GLM53_MTP_TTNN_CACHE=/tmp/glm53_4x4_mtp_cache
  "$HELPER" --timeout-s 800 -k "seq5120 and glm53" $T/test_prefill_block.py::test_glm_prefill_block || return $?
}

section_subtorus_glm_mtp() {
  export MESH_DEVICE=TG
  export GLM53_HF_MODEL=/mnt/weka/model-weights/llm/zai-org/GLM-5.3-fp8-aca966e4
  export TT_GLM53_PREFILL_TTNN_CACHE=/tmp/glm53_4x4_prefill_cache
  export TT_GLM53_MTP_TTNN_CACHE=/tmp/glm53_4x4_mtp_cache
  "$HELPER" --timeout-s 1200 -k "not multiturn and not trunk_tail and (fused or pretrained)" $T/mtp_prefill/test_mtp.py || return $?
  "$HELPER" --timeout-s 900 -k "pretrained and levels1 and aligned_min" $T/mtp_prefill/test_mtp.py::test_mtp_multiturn_pcc || return $?
}

section_subtorus_glm_prefill_transformer() {
  export MESH_DEVICE=TG
  export GLM53_HF_MODEL=/mnt/weka/model-weights/llm/zai-org/GLM-5.3-fp8-aca966e4
  export TT_GLM53_PREFILL_TTNN_CACHE=/mnt/weka/model-cache/scratch/zai-org/GLM-5.3-Cache/GLM-5.3-Cache-prefill
  "$HELPER" --timeout-s 1200 -k "5_layers" $T/test_prefill_transformer.py::test_glm_build_ttnn_cache || return $?
  "$HELPER" --timeout-s 1500 -k "5_layers and smoke and json_prompts and right_pad" $T/test_prefill_transformer.py::test_glm_prefill_transformer || return $?
}

section_subtorus_glm_chunked() {
  export MESH_DEVICE=TG
  export GLM53_HF_MODEL=/mnt/weka/model-weights/llm/zai-org/GLM-5.3-fp8-aca966e4
  export TT_GLM53_PREFILL_TTNN_CACHE=/mnt/weka/model-cache/scratch/zai-org/GLM-5.3-Cache/GLM-5.3-Cache-prefill
  "$HELPER" --timeout-s 1200 -k "L10 and warm_cache and (notrace or traced)" $T/test_prefill_transformer_chunked.py::test_glm_prefill_transformer_chunked || return $?
}

section_subtorus_kimi_mla() {
  export MESH_DEVICE=TG
  "$HELPER" --timeout-s 1500 -k "k2_7 and ((maxedge-1u and cpu and no_determinism) or (plain-5k and func and with_determinism and scalar))" $T/test_mla.py::test_mla_chunked_prefill || return $?
}

section_subtorus_glm_moe_perf() {
  "$HELPER" --timeout-s 900 --expect 6 -k "glm53" $T/perf/test_moe_block_host_perf.py \
    $T/moe_block/test_moe_block_models.py::test_moe_block_perf || return $?
}

section_subtorus_kimi_dflash() {
  export MESH_DEVICE=TG
  export DFLASH_HF_MODEL=/mnt/weka/model-weights/llm/moonshotai/Kimi-K2.7-Code-DFlash
  "$HELPER" --timeout-s 800 -k "pretrained and ctx5k-1chunk" $T/dflash_prefill/test_dflash.py::test_dflash_pcc || return $?
  "$HELPER" --timeout-s 800 -k "pretrained and (aligned_min or midchip_straddle or lastchip or rot_partial)" $T/dflash_prefill/test_dflash.py::test_dflash_multiturn_pcc || return $?
}

list_sections() {
  local entry
  for entry in "${SECTIONS[@]}"; do
    IFS='|' read -r name seconds what <<<"$entry"
    printf '%-36s ~%5ss  %s\n' "$name" "$seconds" "$what"
  done
}

keep_going=false
selected=()
while (($#)); do
  case "$1" in
    --list) list_sections; exit 0 ;;
    --keep-going) keep_going=true ;;
    -*) echo "unknown option: $1" >&2; exit 2 ;;
    *) selected+=("$1") ;;
  esac
  shift
done
if ((${#selected[@]} == 0)); then
  for entry in "${SECTIONS[@]}"; do selected+=("${entry%%|*}"); done
fi

status=0
for name in "${selected[@]}"; do
  if ! declare -F "section_$name" >/dev/null; then
    echo "unknown section: $name (see --list)" >&2
    exit 2
  fi
  echo "===== $name ====="
  start=$SECONDS
  (set -e; "section_$name")
  rc=$?
  echo "===== $name: rc=$rc, $((SECONDS - start)) s ====="
  if ((rc != 0)); then
    status=1
    $keep_going || exit "$status"
  fi
done
exit "$status"
