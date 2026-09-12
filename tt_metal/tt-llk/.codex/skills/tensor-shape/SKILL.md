---
name: tensor-shape
description: >-
  Convert LLK lib/API tile-size args to ckernel::TensorShape and maintain TRISC TensorShape coverage. Use when adding TensorShape parameters, replacing face_r_dim/num_faces, editing LLK_VALIDATE_TENSOR_SHAPE_*, regenerating tensor_shape_coverage_*.h, or reviewing TensorShape PRs.
---

# Canonical Claude Skill

From the TT-LLK repository root, read
`.claude/skills/tensor-shape/SKILL.md` completely and follow its Markdown body
as the authoritative workflow. Ignore only its Claude-specific frontmatter,
including its argument hint. Resolve its relative paths from
`.claude/skills/tensor-shape/` and apply the host mappings in `AGENTS.md`.
