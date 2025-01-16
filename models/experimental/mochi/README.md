# Mochi on TT

This folder contains Tenstorrent's implementation of Mochi on Wormhole. The only supported system for now is T3000.

## Running

First, clone a fork of Genmo's reference repository.
```bash
git clone https://github.com/cglagovichTT/tt-mochi/tree/cglagovich/dev
```
Follow instructions in the the [tt-mochi README](https://github.com/cglagovichTT/tt-mochi/blob/cglagovich/dev/README.md) to download the weights.
```bash
cd tt-mochi
python3 ./scripts/download_weights.py weights/
cd ..
```
Set up your tt-metal environment **outside of tt-mochi** by following the instructions in [INSTALLING](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md).

Create a symlink of the mochi source into tt-metal(TODO: remove this step)
```bash
cd $TT_METAL_HOME
ln -s /path/to/tt-mochi/src/genmo genmo
```
Now run a test:
```bash
export FAKE_DEVICE=T3K # Tells our Mochi to configure itself for T3K
export MOCHI_DIR=/path/to/tt-mochi/weights # Tells our Mochi where to find the weights. This dir should contain weights for the dit, decoder, and encoder.

# Run a DiT test with 1 layer
pytest models/experimental/mochi/tests/test_tt_asymm_dit_joint.py::test_tt_asymm_dit_joint_inference -k "L1"
```
Now you can try generating a video. With those environment variables still set, run
```bash
python models/experimental/mochi/cli_tt.py --model_dir /proj_sw/mochi-data/
```
Grab a few coffees while you wait for the video.
