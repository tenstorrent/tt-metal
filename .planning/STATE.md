# Project State

## Current Position
- **Milestone:** v1.0 complete
- **Status:** Milestone shipped 2026-03-11
- **Next:** Plan next milestone or add phases

## Project Reference

**Core value:** Transparent auto-packetization for fabric APIs — callers send any size, chunking is invisible
**Current focus:** Milestone complete — ready for next milestone or additional work

## Decisions
- _single_packet suffix for renamed APIs; wrappers keep original names
- Breadth-first multi-connection chunking
- Scatter wrappers are passthrough (pre-computed NOC addresses)
- mesh SetRoute=false pattern
- Fused ops: intermediate chunks as regular writes, atomic_inc only on final chunk
- Tests compiled into fabric_unit_tests binary
- detail::CompileProgram for device kernel compile-only validation

## Blockers
None

## Last Session
- **Timestamp:** 2026-03-11
- **Stopped At:** Milestone v1.0 complete
