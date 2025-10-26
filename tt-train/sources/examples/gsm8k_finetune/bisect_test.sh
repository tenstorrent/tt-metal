git sumbmodule update --init --recursive
../../../../build_metal.sh -b Release
cmake --build ~/git/tt-metal/tt-train/build --config Release --target all
python3 gsm8k_finetune.py
exit $?