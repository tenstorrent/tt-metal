# Shared environment for the SP fusion work (durable copy under generated/, git-ignored). Source from bash.
export TT_METAL_HOME=/home/imichalak/tenstorrent/tt-metal
export TT_METAL_RUNTIME_ROOT=/home/imichalak/tenstorrent/tt-metal
export SPFUSE=/home/imichalak/tenstorrent/tt-metal/generated/spfuse
export PY=$TT_METAL_HOME/python_env/bin/python
export MGD_1x4_RING=$SPFUSE/mgd/bh_galaxy_1_4_ring_ring.textproto
export MGD_1x4_LINE=$SPFUSE/mgd/bh_galaxy_1_4_line_line.textproto
export MGD_1x2_LINE=$TT_METAL_HOME/tt-train/configs/mgd/bh_galaxy_1_2_line_line.textproto
ulimit -u 65536 2>/dev/null || true
