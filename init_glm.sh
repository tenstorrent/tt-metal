#!/usr/bin/env bash
# Source before launching Claude Code:  source init_glm.sh
export ANTHROPIC_DEFAULT_OPUS_MODEL="zai-org/glm-5.2"
export ANTHROPIC_DEFAULT_SONNET_MODEL="zai-org/glm-5.2"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="zai-org/glm-5.2"
export CLAUDE_CODE_AUTO_COMPACT_WINDOW="1000000"
export CLAUDE_CODE_SUBAGENT_MODEL="zai-org/glm-5.2"
export ANTHROPIC_BASE_URL="https://api.aiand.com"
# GLM endpoint caps a single response at 32K output tokens; the implementer
# agent (writing a full SDPA op + kernels) blows past that and dies with
# "Claude's response exceeded the 32000 output token maximum". Raise to 64K.
export CLAUDE_CODE_MAX_OUTPUT_TOKENS=64000
# Pin pipeline subprocesses to max effort. The pipeline's run_claude reads
# CLAUDE_EFFORT_LEVEL and passes --effort <level> to every claude -p (fresh +
# resume), serialized to output_config.effort on the wire. Verified: max ->
# output_config.effort:"max". Overrides the interactive CLAUDE_EFFORT for runs.
export CLAUDE_EFFORT_LEVEL=max
export ANTHROPIC_API_KEY=
# JIT compile farm — offloads kernel compilation to bgdepyc01. Sourced by every
# run_eval/run_refinements/run_op invocation, so the endpoint is always set.
export TT_METAL_JIT_SERVER_ENDPOINT=bgdepyc01:54778
