# t232 job env (blx01): tree b (t228 key-phase code b95861393fe, own build). No overlay.
F=/var/tmp/fasth3; T=$F/t232; B=$T/b
mkdir -p $F/tmp
export HOME=$F/home XDG_CACHE_HOME=$F/home/.cache TMPDIR=$F/tmp TORCH_HOME=$F/home/.cache/torch HF_HOME=$F/home/.cache/huggingface
source $F/t48/python_env/bin/activate
export TT_METAL_HOME=$B PYTHONPATH=$B:$B/ttnn:$B/tools HF_HUB_OFFLINE=1
export TT_METAL_CACHE=$F/cache/tt-metal-cache
