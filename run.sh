kill -9 $(pgrep -f "python test.py")
kill -9 $(pgrep -f "python test.py")


export TT_METAL_DPRINT_ONE_FILE_PER_RISC=1
export TT_METAL_DPRINT_CORES=0,6
# export TT_METAL_DPRINT_CORES=all

tt-smi -r 0
python test.py
