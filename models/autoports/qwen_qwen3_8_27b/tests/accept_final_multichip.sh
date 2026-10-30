#!/usr/bin/env bash
set -euo pipefail
b=models/autoports/qwen_qwen3_8_27b
doc=$b/doc/optimized_multichip_decoder
envscript=$b/tests/optimized_multichip_env.sh
for layer in 0 3; do
 docker exec qwen38-functional-build bash "$envscript" python -m models.autoports.qwen_qwen3_8_27b.tests.run_multichip_decoder --layer "$layer" --seq 4096 --steps 128 --prefill-repeats 15 --output "$doc/final_accepted_l$layer.json" > "$doc/final_accepted_l$layer.log" 2>&1
done
for suite in batch boundary continuation context stack watcher; do
 docker exec qwen38-functional-build bash "$envscript" python -m models.autoports.qwen_qwen3_8_27b.tests.validate_optimized_multichip --suite "$suite" > "$doc/${suite}_validation.log" 2>&1
done
docker exec qwen38-functional-build bash "$envscript" python -m models.autoports.qwen_qwen3_8_27b.tests.probe_packed_decode_conv --output "$doc/packed_conv_exact.json" > "$doc/packed_conv_exact.log" 2>&1
for probe in packed_rope prefill_rope; do
 docker exec qwen38-functional-build bash "$envscript" python -m models.autoports.qwen_qwen3_8_27b.tests.probe_$probe --output "$doc/final_${probe}_exact.json" > "$doc/final_${probe}_exact.log" 2>&1
done
docker exec qwen38-functional-build bash "$envscript" python -m models.autoports.qwen_qwen3_8_27b.tests.probe_decode_workspace --output "$doc/final_workspace_lifetime.json" > "$doc/final_workspace_lifetime.log" 2>&1
for layer in 0 3; do
 docker exec qwen38-functional-build bash "$envscript" python -m models.autoports.qwen_qwen3_8_27b.tests.probe_mixed_decoder_traces --layer "$layer" --output "$doc/mixed_traces_l$layer.json" > "$doc/mixed_traces_l$layer.log" 2>&1
done

if [[ ! -f "$doc/mixed_prefill_decode_traces.json" ]]; then
 docker exec qwen38-functional-build bash "$envscript" python -m models.autoports.qwen_qwen3_8_27b.tests.probe_mixed_prefill_decode_traces --output "$doc/mixed_prefill_decode_traces.json" > "$doc/mixed_prefill_decode_traces.log" 2>&1
fi
for layer in 0 3; do
 docker exec qwen38-functional-build bash "$envscript" python -m models.autoports.qwen_qwen3_8_27b.tests.run_multichip_decoder --layer "$layer" --seq 262142 --steps 2 --l1-background-bytes 2097152 --output "$doc/final_reserved_context_l$layer.json" > "$doc/final_reserved_context_l$layer.log" 2>&1
done
docker exec qwen38-functional-build bash "$envscript" python -m models.autoports.qwen_qwen3_8_27b.tests.run_multichip_decoder --layer 3 --batch 8 --seq 4097 --steps 8 --output "$doc/packed_prefill_multichunk_b8_l3.json" > "$doc/packed_prefill_multichunk_b8_l3.log" 2>&1
for layer in 0 3; do
 docker exec qwen38-functional-build bash "$envscript" bash "$b/tests/profile_optimized_multichip.sh" "final_profile_l$layer" --layer "$layer" --multichip-only > "$doc/final_profile_l$layer.log" 2>&1
done
for layer in 0 3; do
 docker exec qwen38-functional-build bash "$envscript" bash "$b/tests/profile_optimized_multichip.sh" "final_b32_profile_l$layer" --layer "$layer" --batch 32 --multichip-only > "$doc/final_b32_profile_l$layer.log" 2>&1
done
