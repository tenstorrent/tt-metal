# Refuted forced-deallocation hypothesis

CPU source audit only; no hardware experiment or implementation change.
The coordinator initially inspected `models/demos/gpt_oss/config.py::allreduce`,
which force-deallocates its input before AG allocation and its scattered tensor
after AG dispatch. That is not the MeshConfig imported by this decoder.

The actual import at `tt/multichip_decoder.py:25` is
`models.demos.gemma4.config.MeshConfig`. Its allreduce implementation retains
natural local references through both collectives and has no forced deallocation
on the active unpadded path. Only its optional padding branch deallocates;
`pad_size=None` here. Only CCLManager comes from GPT-OSS. The proposed no-force-
deallocation control would therefore make no relevant production change and
was rejected before execution. Do not use the other helper as evidence of a
lifetime bug in this model.

The outer-retaining diagnostic still changes allocator lifetimes, so its pass
is not equivalent to the ordinary failing run. However the original-class
TP4-only output diagnostic also passes128. Investigate the concrete difference
between the ordinary paired TP1→TP4 runner and TP4-only diagnostic before
attributing the failure to retained collective buffers.
