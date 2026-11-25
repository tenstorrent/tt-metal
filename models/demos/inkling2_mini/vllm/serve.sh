#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
#
# Inkling-2-Mini behind vLLM (upstream vLLM + tenstorrent/vllm-tt-plugin): `vllm serve` with this model's arguments.
#
#   INKLING_CKPT=<checkpoint dir> models/demos/inkling2_mini/vllm/serve.sh [more vllm serve arguments]
#
# Run it in the python env that has ttnn, vLLM and the plugin (see the README, "Serve with vLLM"). It holds the
# card until Ctrl-C. Environment:
#   INKLING_CKPT       the checkpoint directory: tokenizer and chat template (required)
#   HOST, PORT         where it listens (127.0.0.1, 8000). HOST=0.0.0.0 opens it to the network: add --api-key KEY
#   CTX                --max-model-len, the context the far KV cache is allocated for (196608)
#   SPEC=0             no speculative decode (it is on: greedy and sampled requests; needs the drafter weights)
#   FAST=1             fast mode (own3-l8 experts)
#   PREFILL=<tokens>   chunked prefill for new prompts of at least that many tokens (128). PREFILL=0 or PREFILL=
#                      (empty) turns it off: every prompt is fed token by token. A reply after a prefilled prompt
#                      comes from the prefill's numerics. It needs DRAM: with speculative decode above CTX=131072
#                      (so in the default launch) it stays off; the server says at start what it uses or lacks
# and the model's own: INKLING_MODE, INKLING_CACHE_DIR, INKLING_CACHE, INKLING_SPEC_MAX_CTX (modes.py).
set -e
: "${INKLING_CKPT:?set INKLING_CKPT to the checkpoint directory (tokenizer, chat template)}"
cd "$(dirname "$0")/../../../.."

class=Inkling2SpecForCausalLM
trace=536870912     # the draft, verify, commit and stash traces take about 270 MiB
ring='"worker_l1_size": 1394752, '      # the 64 KB kernel-config ring (tt/device.py)
spec=(--no-async-scheduling --speculative-config
      '{"method": "custom_class", "model": "vllm_tt_plugin.model_owned_drafter", "num_speculative_tokens": 2}')
PREFILL=${PREFILL-128}  # (set and empty: off)
# shortcut: speculative decode above a 131072 context leaves too little DRAM for the prefill, and the prefill's
# 48 KB ring would cost such a launch decode speed for nothing (70 against 77 tok/s): it stays off there.
# Upgrade when the prefill's DRAM need drops.
if [ "${SPEC:-1}" != 0 ] && [ "${CTX:-196608}" -gt 131072 ] && [ -n "$PREFILL" ] && [ "$PREFILL" != 0 ]; then
    echo "serve.sh: chunked prefill off: speculative decode at CTX=${CTX:-196608} leaves too little DRAM" \
         "(SPEC=0, or CTX=131072 or less, turns it on)" >&2
    PREFILL=0
fi
if [ -n "$PREFILL" ] && [ "$PREFILL" != 0 ]; then
    export INKLING_VLLM_PREFILL=$PREFILL
    ring='"worker_l1_size": 1411136, '  # 48 KB: the prefill's largest program needs the rest
fi
if [ "${SPEC:-1}" = 0 ]; then
    class=Inkling2ForCausalLM
    trace=134217728
    spec=()
fi
export TT_MODEL_CLASS_OVERRIDES=Inkling2ForCausalLM=models.demos.inkling2_mini.tt.generator_vllm:$class

exec vllm serve models/demos/inkling2_mini/vllm --tokenizer "$INKLING_CKPT" --served-model-name Inkling-2-Mini \
    --max-model-len "${CTX:-196608}" --max-num-seqs 1 --shutdown-timeout 60 \
    --reasoning-parser inkling --enable-auto-tool-choice --tool-call-parser inkling \
    --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}" "${spec[@]}" \
    --additional-config "{\"tt\": {$ring\"trace_region_size\": $trace, \"sample_on_device_mode\": \"all\"${FAST:+, \"optimizations\": \"performance\"}}}" \
    "$@"
