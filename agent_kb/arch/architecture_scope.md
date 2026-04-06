---
title: Architecture Scope
type: arch
status: seed
confidence: medium
last_reviewed: 2026-04-06
tags:
  - architecture
  - wormhole
  - blackhole
source_files:
  - docs/source/tt-metalium/get_started/get_started.rst
  - docs/source/tt-metalium/tt_metal/advanced_topics/compute_engines_and_dataflow_within_tensix.rst
  - README.md
---

# Architecture Scope

## Why This Page Exists

Agent guidance becomes dangerous when hardware-generation details are flattened into one universal rule.

The KB should record generation-specific behavior explicitly instead of letting those differences leak into generic pages.

## Current Policy

- If a rule is known to be general, keep it in a concept or recipe page.
- If a rule depends on hardware generation, put the scoped detail here or in a dedicated subpage and link to it.
- If the scope is uncertain, mark it as uncertain rather than generalizing.

## Near-Term Targets

As the KB grows, add scoped notes for:

- Wormhole vs Blackhole accumulation and register assumptions
- architecture-specific examples worth preferring during codegen
- hardware-specific debugging heuristics

## Current Confidence

This page is intentionally conservative. It is a placeholder for future architecture-specific guidance rather than a complete map of differences.

## Sources

- `docs/source/tt-metalium/get_started/get_started.rst`
- `docs/source/tt-metalium/tt_metal/advanced_topics/compute_engines_and_dataflow_within_tensix.rst`
- `README.md`
