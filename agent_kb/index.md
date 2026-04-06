# Agent KB Index

## Concepts

- [Kernel Pipeline](concepts/kernel_pipeline.md): Reader, compute, writer, and the control-flow boundaries agents must preserve.
- [Circular Buffers](concepts/circular_buffers.md): Producer-consumer semantics and the core reserve/push/wait/pop protocol.

## Pitfalls

- [Kernel Codegen Gotchas](pitfalls/kernel_codegen_gotchas.md): High-frequency mistakes around synchronization, accuracy, shapes, and API assumptions.
- [Conv3D Gotchas](pitfalls/conv3d_gotchas.md): Op-specific traps around weight prep, reduction, blocking, and layout support.

## Recipes

- [New Kernel Codegen Checklist](recipes/new_kernel_codegen_checklist.md): Minimum process for generating or modifying kernels safely.
- [Conv3D Codegen](recipes/conv3d_codegen.md): What to inspect together before changing the current `conv3d` op.

## Debug Playbooks

- [Hangs and Runtime Triage](debug_playbooks/hangs_and_runtime_triage.md): How to use Watcher, DPRINT, and compile-time inspection when kernels misbehave.

## Architecture Notes

- [Architecture Scope](arch/architecture_scope.md): How to record hardware-generation-specific guidance without over-generalizing.

## Source Summaries

- [Bootstrap Sources](sources/bootstrap_sources.md): Initial high-value source set for the KB.
- [Conv3D Operation](sources/conv3d_op.md): Current structure, config model, and test-backed constraints for `ttnn.experimental.conv3d`.
