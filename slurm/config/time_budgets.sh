#!/usr/bin/env bash
# time_budgets.sh - Timeout values for Slurm --time directives
# Derived from .github/time_budget.yaml. Values in minutes.

[[ -n "${_SLURM_CI_TIME_BUDGETS_SH:-}" ]] && return 0
_SLURM_CI_TIME_BUDGETS_SH=1

# Usage: get_time_budget <team> <pipeline> <sku>
# Returns timeout in minutes, or the default if not found.
get_time_budget() {
    local team="$1" pipeline="$2" sku="$3"
    local key="${team}_${pipeline}_${sku}"
    local default="${DEFAULT_JOB_TIMEOUT_MINUTES:-120}"

    # Lookup table (team_pipeline_sku -> minutes)
    case "${key}" in
        # ttnn
        ttnn_sanity_wh_n300_civ2)   echo 535 ;;
        ttnn_sanity_bh_p100)        echo 535 ;;
        ttnn_sanity_bh_p150b_civ2)  echo 535 ;;
        ttnn_unit_wh_llmbox)        echo 50 ;;
        ttnn_e2e_wh_llmbox)         echo 90 ;;
        ttnn_e2e_wh_galaxy)         echo 40 ;;
        ttnn_e2e_bh_galaxy)         echo 40 ;;
        ttnn_integration_wh_llmbox) echo 30 ;;
        ttnn_integration_wh_galaxy) echo 350 ;;

        # fabric
        fabric_unit_wh_llmbox)        echo 40 ;;
        fabric_integration_wh_llmbox) echo 10 ;;
        fabric_e2e_wh_llmbox)         echo 165 ;;
        fabric_e2e_wh_galaxy)         echo 100 ;;
        fabric_e2e_bh_galaxy)         echo 60 ;;

        # scaleout
        scaleout_e2e_wh_llmbox)         echo 20 ;;
        scaleout_unit_wh_llmbox)        echo 20 ;;
        scaleout_integration_wh_llmbox) echo 30 ;;

        # models
        models_e2e_wh_llmbox)            echo 180 ;;
        models_e2e_wh_llmbox_perf)       echo 60 ;;
        models_e2e_wh_n150)              echo 30 ;;
        models_e2e_wh_n300)              echo 30 ;;
        models_e2e_wh_galaxy)            echo 30 ;;
        models_unit_wh_llmbox)           echo 305 ;;
        models_unit_wh_llmbox_perf)      echo 60 ;;
        models_unit_wh_n150)             echo 30 ;;
        models_unit_wh_n300)             echo 30 ;;
        models_unit_wh_galaxy)           echo 30 ;;
        models_integration_wh_llmbox)    echo 422 ;;
        models_perf_wh_llmbox_perf)      echo 685 ;;
        models_perf_wh_galaxy)           echo 280 ;;
        models_demo_wh_llmbox_perf)      echo 1090 ;;
        models_demo_wh_llmbox_perf_release) echo 1090 ;;
        models_demo_wh_galaxy)           echo 287 ;;
        models_demo_wh_galaxy_release)   echo 345 ;;
        models_demo_wh_n150)             echo 500 ;;
        models_demo_wh_n150_civ2)        echo 300 ;;
        models_demo_wh_n300)             echo 600 ;;
        models_demo_wh_n300_civ2)        echo 400 ;;
        models_demo_wh_n300_perf)        echo 200 ;;
        models_demo_bh_p150)             echo 250 ;;
        models_demo_bh_p150_perf)        echo 420 ;;
        models_demo_bh_deskbox)          echo 250 ;;
        models_demo_bh_llmbox)           echo 1200 ;;
        models_demo_bh_loudbox)          echo 1200 ;;
        models_demo_bh_p300)             echo 800 ;;
        models_demo_bh_quietbox_2)       echo 250 ;;
        models_sweep_wh_n150)            echo 15 ;;
        models_sweep_wh_llmbox_perf)     echo 30 ;;

        *) echo "${default}" ;;
    esac
}

# Format minutes as HH:MM:SS for #SBATCH --time
minutes_to_slurm_time() {
    local mins="$1"
    local hours=$((mins / 60))
    local remaining=$((mins % 60))
    printf '%02d:%02d:00' "${hours}" "${remaining}"
}
