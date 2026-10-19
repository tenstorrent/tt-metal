# Common setup for the #59846 SDPA sweep on bh-glx-120-c04u02.
cd /home/mmanzoor/compiler/tt-metal
source activate
export D=/data/mmanzoor/sdpa_chunk_sweep_c04u02
export T=tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py
export GEMMA4_SDPA_SWEEP_REPEATS=${GEMMA4_SDPA_SWEEP_REPEATS:-6}
