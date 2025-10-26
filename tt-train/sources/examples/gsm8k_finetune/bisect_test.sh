git sumbmodule update --init --recursive
build_metal.sh -b Release
cmake --build ~/git/tt-metal/tt-train/build --config Release --target all
python3 tt-train/sources/examples/gsm8k_finetune/gsm8k_finetune.py
exit $?