# t227 job env (blx01): tree b (t212, 8b1167ef43b, own build) with the t219 overlay ov first on PYTHONPATH.
F=/var/tmp/fasth3; T=$F/t227; OV=$F/t219/ov; B=$F/t212/b
mkdir -p $F/tmp
export HOME=$F/home XDG_CACHE_HOME=$F/home/.cache TMPDIR=$F/tmp TORCH_HOME=$F/home/.cache/torch HF_HOME=$F/home/.cache/huggingface
source $F/t48/python_env/bin/activate
export TT_METAL_HOME=$B PYTHONPATH=$OV:$B:$B/ttnn:$B/tools HF_HUB_OFFLINE=1
export TT_METAL_CACHE=$F/cache/tt-metal-cache
