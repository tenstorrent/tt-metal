# Source this file to run tt-metal / ttnn on the craq-sim Quasar simulator instead of real hardware:
#
#   . quasar_practice/craq_setup.sh
#
# On a fresh machine it clones craq-sim, builds the Quasar simulator and stages it; afterwards it
# only exports the variables. The variables are the ones tt-metal CI sets for simulator runs
# (.github/actions/setup-ttsim/action.yml).

export TT_METAL_HOME=$HOME/tt-metal
CRAQ_SIM_HOME=$HOME/craq-sim          # craq-sim checkout, branch quasar
QSR_SIM_DIR=$HOME/sim/qsr             # libttsim.so + soc_descriptor.yaml, staged together

# --- one-time: clone, build, stage ----------------------------------------------------------------
if [[ ! -d $CRAQ_SIM_HOME ]]; then
    git clone --branch quasar https://github.com/tenstorrent/craq-sim.git "$CRAQ_SIM_HOME" || return 1
fi
if [[ ! -f $CRAQ_SIM_HOME/src/_out/release_qsr/libttsim.so ]]; then
    # TT_VERSION 0/1/2 = Wormhole/Blackhole/Quasar
    (cd "$CRAQ_SIM_HOME" && TT_VERSION=2 ./make.py src/_out/release_qsr/libttsim.so) || return 1
fi
# tt-metal needs libttsim.so and soc_descriptor.yaml in one directory. The _ttsim descriptor is
# quasar_32_arch.yaml without the dispatch engines ttsim does not model. cp -u refreshes the staged
# copies after a craq-sim rebuild.
mkdir -p "$QSR_SIM_DIR"
cp -u "$CRAQ_SIM_HOME/src/_out/release_qsr/libttsim.so" "$QSR_SIM_DIR/libttsim.so"
cp -u "$TT_METAL_HOME/tt_metal/soc_descriptors/quasar_32_arch_ttsim.yaml" "$QSR_SIM_DIR/soc_descriptor.yaml"

# --- what makes tt-metal talk to the simulator instead of a chip ----------------------------------
export TT_METAL_SIMULATOR_HOME=$QSR_SIM_DIR
export TT_METAL_SIMULATOR=$QSR_SIM_DIR/libttsim.so

# The simulator does not model the fast-dispatch engines, so the host writes programs directly.
export TT_METAL_SLOW_DISPATCH_MODE=1

# Set by tt-metal CI for every simulator run.
export TT_METAL_DISABLE_SFPLOADMACRO=1
# Old NoC API path; the published Quasar simulator has a known gap with V2 (tt-metal #54794).
export TT_METAL_QUASAR_NOC_API_VERSION=1
