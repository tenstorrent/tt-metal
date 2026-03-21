#!/usr/bin/env bash
# Maps SKU names (from .github/sku_config.yaml) to Slurm partition/constraint flags.
# Source this file, then call: get_slurm_args <sku_name>
set -euo pipefail

# get_slurm_args <sku_name>
#   Prints shell variable assignments for SLURM_PARTITION and SLURM_CONSTRAINT.
#   Caller should eval the output.
#   Returns 1 if the SKU is unknown.
#
# Example:
#   eval "$(get_slurm_args wh_llmbox_perf)"
#   sbatch ${SLURM_PARTITION} ${SLURM_CONSTRAINT} job.sh
get_slurm_args() {
    local sku="${1:?Usage: get_slurm_args <sku_name>}"
    local partition=""
    local constraint=""

    case "${sku}" in
        # -----------------------------------------------------------------
        # Wormhole single-card
        # -----------------------------------------------------------------
        wh_n150)
            partition="wh-n150"
            ;;
        wh_n150_civ2)
            partition="wh-n150"
            constraint="viommu"
            ;;
        wh_n300)
            partition="wh-n300"
            ;;
        wh_n300_civ2)
            partition="wh-n300"
            constraint="viommu"
            ;;
        wh_n300_perf)
            partition="wh-n300"
            constraint="perf"
            ;;

        # -----------------------------------------------------------------
        # Wormhole T3000 (LLMBox)
        # -----------------------------------------------------------------
        wh_llmbox)
            partition="wh-t3k"
            ;;
        wh_llmbox_perf|wh_llmbox_perf_release)
            partition="wh-t3k"
            constraint="perf"
            ;;
        wh_llmbox_civ2)
            partition="wh-t3k"
            constraint="viommu"
            ;;

        # -----------------------------------------------------------------
        # Wormhole Galaxy (6U topology)
        # -----------------------------------------------------------------
        wh_galaxy|wh_galaxy_release)
            partition="wh-galaxy"
            ;;

        # -----------------------------------------------------------------
        # Blackhole P-series
        # -----------------------------------------------------------------
        bh_p100)
            partition="bh-p100"
            ;;
        bh_p100_perf)
            partition="bh-p100"
            constraint="perf"
            ;;
        bh_p100a_civ2_viommu)
            partition="bh-p100"
            constraint="viommu"
            ;;
        bh_p150|bh_p150b_civ2)
            partition="bh-p150"
            ;;
        bh_p150_perf)
            partition="bh-p150"
            constraint="perf"
            ;;
        bh_p150b_civ2_viommu)
            partition="bh-p150"
            constraint="viommu"
            ;;
        bh_p300)
            partition="bh-p300"
            ;;
        bh_p300_viommu)
            partition="bh-p300"
            constraint="viommu"
            ;;

        # -----------------------------------------------------------------
        # Blackhole multi-chip systems
        # -----------------------------------------------------------------
        bh_llmbox)
            partition="bh-llmbox"
            ;;
        bh_loudbox)
            partition="bh-loudbox"
            ;;
        bh_loudbox_civ2_viommu)
            partition="bh-loudbox"
            constraint="viommu"
            ;;
        bh_deskbox)
            partition="bh-deskbox"
            ;;
        bh_quietbox|bh_quietbox_perf)
            partition="bh-llmbox"
            constraint="quietbox"
            ;;
        bh_quietbox_2)
            partition="bh-llmbox"
            constraint="qb-ge"
            ;;

        # -----------------------------------------------------------------
        # Blackhole Galaxy
        # -----------------------------------------------------------------
        bh_galaxy)
            partition="bh-galaxy"
            ;;

        # -----------------------------------------------------------------
        # CPU-only / build
        # -----------------------------------------------------------------
        CPU|cpu)
            partition="build"
            ;;

        *)
            echo "ERROR: Unknown SKU '${sku}'" >&2
            return 1
            ;;
    esac

    echo "SLURM_PARTITION='--partition=${partition}'"
    if [[ -n "${constraint}" ]]; then
        echo "SLURM_CONSTRAINT='--constraint=${constraint}'"
    else
        echo "SLURM_CONSTRAINT=''"
    fi
}

# get_partition <sku_name>
#   Convenience: returns just the partition name (no flags).
get_partition() {
    local sku="${1:?Usage: get_partition <sku_name>}"
    local output
    output="$(get_slurm_args "${sku}")" || return 1
    eval "${output}"
    echo "${SLURM_PARTITION#--partition=}"
}

# get_arch_name <sku_name>
#   Returns the ARCH_NAME for a given SKU (wormhole_b0 or blackhole).
get_arch_name() {
    local sku="${1:?Usage: get_arch_name <sku_name>}"
    case "${sku}" in
        wh_*) echo "wormhole_b0" ;;
        bh_*) echo "blackhole" ;;
        *)    echo "unknown" ;;
    esac
}

# is_viommu_sku <sku_name>
#   Returns 0 (true) if the SKU requires viommu / CIv2 support.
is_viommu_sku() {
    local sku="${1:?Usage: is_viommu_sku <sku_name>}"
    case "${sku}" in
        *_civ2*|*_viommu*) return 0 ;;
        *)                 return 1 ;;
    esac
}
