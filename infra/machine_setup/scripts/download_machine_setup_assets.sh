#!/usr/bin/env bash

set -eo pipefail

TT_METAL_HOME=$(git rev-parse --show-toplevel)
ASSETS_DIR=$TT_METAL_HOME/infra/machine_setup/assets

SERVER=yyz-lab-65
GS_TT_SMI_SERVER_LOCATION=/home/software/syseng/bin/tt-smi/gs/tt-smi_2022-12-05-74801e089fb2e564
WH_TT_SMI_SERVER_LOCATION=/home/software/syseng/bin/tt-smi/wh/tt-smi-wh_2023-02-07-eda6cb21c5788763
GS_TT_SMI_LOCAL_FOLDER=$ASSETS_DIR/tt_smi/gs
WH_TT_SMI_LOCAL_FOLDER=$ASSETS_DIR/tt_smi/wh
GS_TT_SMI_LOCAL_LOCATION=$GS_TT_SMI_LOCAL_FOLDER/tt-smi
WH_TT_SMI_LOCAL_LOCATION=$WH_TT_SMI_LOCAL_FOLDER/tt-smi

rm -rf $ASSETS_DIR
mkdir -p $GS_TT_SMI_LOCAL_FOLDER
mkdir -p $WH_TT_SMI_LOCAL_FOLDER

scp $SERVER:"$GS_TT_SMI_SERVER_LOCATION" $GS_TT_SMI_LOCAL_LOCATION
scp $SERVER:"$WH_TT_SMI_SERVER_LOCATION" $WH_TT_SMI_LOCAL_LOCATION
