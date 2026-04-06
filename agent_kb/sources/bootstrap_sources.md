---
title: Bootstrap Sources
type: source
status: seed
confidence: high
last_reviewed: 2026-04-06
tags:
  - sources
  - bootstrap
source_files:
  - README.md
  - METALIUM_GUIDE.md
  - CONTRIBUTING.md
  - docs/source/tt-metalium/get_started/get_started.rst
  - docs/source/tt-metalium/tt_metal/advanced_topics/compute_engines_and_dataflow_within_tensix.rst
  - docs/source/tt-metalium/tt_metal/apis/kernel_apis/circular_buffers/circular_buffers.rst
  - tech_reports/op_kernel_dev/accuracy_tips/accuracy_tips.md
  - tt_metal/programming_examples
---

# Bootstrap Sources

This page records the first raw sources the KB should rely on most heavily.

## Core Sources

- `README.md`
  - Good top-level map of docs, tech reports, examples, and tools.
- `METALIUM_GUIDE.md`
  - Best single-file overview of architecture, kernel roles, and example pipeline structure.
- `docs/source/tt-metalium/get_started/get_started.rst`
  - Good progression for how developers are expected to learn the system.
- `docs/source/tt-metalium/tt_metal/advanced_topics/compute_engines_and_dataflow_within_tensix.rst`
  - Critical for Dst register ownership, engine/register flow, and compute-loop sequencing.
- `docs/source/tt-metalium/tt_metal/apis/kernel_apis/circular_buffers/circular_buffers.rst`
  - Minimal but important API inventory for CB semantics.
- `tech_reports/op_kernel_dev/accuracy_tips/accuracy_tips.md`
  - High-value source for subtle numerical and boundary-case behavior.
- `CONTRIBUTING.md`
  - Best source for debugging workflows, Watcher usage, and investigating device-side issues.
- `tt_metal/programming_examples/`
  - Ground truth for current implementation patterns.

## Retrieval Priority

When guiding code generation:

1. Search seeded KB pages.
2. Read the closest programming example.
3. Read the relevant advanced-topic or API page.
4. Read debugging or accuracy guidance if the change touches those areas.

## Sources

- `README.md`
- `METALIUM_GUIDE.md`
- `CONTRIBUTING.md`
- `docs/source/tt-metalium/get_started/get_started.rst`
- `docs/source/tt-metalium/tt_metal/advanced_topics/compute_engines_and_dataflow_within_tensix.rst`
- `docs/source/tt-metalium/tt_metal/apis/kernel_apis/circular_buffers/circular_buffers.rst`
- `tech_reports/op_kernel_dev/accuracy_tips/accuracy_tips.md`
- `tt_metal/programming_examples`
