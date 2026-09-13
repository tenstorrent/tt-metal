"""Torch reference implementation of MiniMax Music 3 (vendored from the diffusers PR #14456 modules, see PROVENANCE.md).

Used for: CPU goldens + teacher-forced dumps (stage 01), PCC references for the TTNN modules, and the CPU parts of
the serving path (condition encoder, vocoder) in phase A.
"""
