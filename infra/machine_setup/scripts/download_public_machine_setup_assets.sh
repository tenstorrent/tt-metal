#!/usr/bin/env bash

set -eo pipefail

if [[ -z "$GITHUB_TOKEN" ]]; then
  echo "Must provide GITHUB_TOKEN in environment" 1>&2
  exit 1
fi

release=false

# Parse command-line arguments
while getopts ":r" opt; do
  case $opt in
    r)
      release=true
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
  esac
done

TT_METAL_HOME=$(git rev-parse --show-toplevel)
ASSETS_DIR=$TT_METAL_HOME/infra/machine_setup/assets

# FULL_REPO_NAME=tenstorrent-software/pybuda
FULL_REPO_NAME=tenstorrent-metal/tt-metal-sys-eng-packages

PYBUDA_GS_RELEASE=v2023.9.19-grayskull
PYBUDA_WH_RELEASE=v2023.9.7-wormhole_b0
GS_TT_SMI_FILENAME=tt-smi_2023-06-16-0283a02404487eea
WH_TT_SMI_FILENAME=tt-smi-8.6.0.0_2023-08-22-492ad2b9ef82a243
GS_TT_FLASH_FILENAME=tt-flash_2023-06-28-91e1cc1ef8caea8f
WH_TT_FLASH_FILENAME=tt-flash_7.D.0.0_2023-08-08-7ab3bd015206a6ff
GS_TT_DRIVER_FILENAME=install_ttkmd_1.26.bash

PYBUDA_GS_RELEASE_ID=$(curl -L -H "Authorization: Bearer $GITHUB_TOKEN" \
	-H "X-GitHub-Api-Version: 2022-11-28" \
	-H "Accept: application/vnd.github+json" \
	https://api.github.com/repos/$FULL_REPO_NAME/releases/tags/$PYBUDA_GS_RELEASE |
	jq '.id')

PYBUDA_GS_RELEASE_ASSETS=$(curl -L -H "Authorization: Bearer $GITHUB_TOKEN" \
	-H "X-GitHub-Api-Version: 2022-11-28" \
	-H "Accept: application/vnd.github+json" \
	https://api.github.com/repos/$FULL_REPO_NAME/releases/$PYBUDA_GS_RELEASE_ID/assets)

PYBUDA_WH_RELEASE_ID=$(curl -L -H "Authorization: Bearer $GITHUB_TOKEN" \
	-H "X-GitHub-Api-Version: 2022-11-28" \
	-H "Accept: application/vnd.github+json" \
	https://api.github.com/repos/$FULL_REPO_NAME/releases/tags/$PYBUDA_WH_RELEASE |
	jq '.id')

PYBUDA_WH_RELEASE_ASSETS=$(curl -L -H "Authorization: Bearer $GITHUB_TOKEN" \
	-H "X-GitHub-Api-Version: 2022-11-28" \
	-H "Accept: application/vnd.github+json" \
	https://api.github.com/repos/$FULL_REPO_NAME/releases/$PYBUDA_WH_RELEASE_ID/assets)

GS_TT_SMI_SERVER_LOCATION=$(echo $PYBUDA_GS_RELEASE_ASSETS | jq ".[] | select(.name==\"$(echo $GS_TT_SMI_FILENAME)\")" | jq '.url' | tr \" \ )
WH_TT_SMI_SERVER_LOCATION=$(echo $PYBUDA_WH_RELEASE_ASSETS | jq ".[] | select(.name==\"$(echo $WH_TT_SMI_FILENAME)\")" | jq '.url' | tr \" \ )
GS_TT_FLASH_SERVER_LOCATION=$(echo $PYBUDA_GS_RELEASE_ASSETS | jq ".[] | select(.name==\"$(echo $GS_TT_FLASH_FILENAME)\")" | jq '.url' | tr \" \ )
WH_TT_FLASH_SERVER_LOCATION=$(echo $PYBUDA_WH_RELEASE_ASSETS | jq ".[] | select(.name==\"$(echo $WH_TT_FLASH_FILENAME)\")" | jq '.url' | tr \" \ )
GS_TT_DRIVER_SERVER_LOCATION=$(echo $PYBUDA_GS_RELEASE_ASSETS | jq ".[] | select(.name==\"$(echo $GS_TT_DRIVER_FILENAME)\")" | jq '.url' | tr \" \ )

mkdir -p $ASSETS_DIR

# We download all assets into a flat list in a directory to upload later
# as release assets with filenames as is. We can't use the generic names
# that are for Ansible installations as we want to preserve filenames,
# especially if version numbers are implanted in them.
if [ "$release" = true ]; then
  GS_TT_SMI_LOCAL_LOCATION=$ASSETS_DIR/$GS_TT_SMI_FILENAME
  WH_TT_SMI_LOCAL_LOCATION=$ASSETS_DIR/$WH_TT_SMI_FILENAME
  GS_TT_FLASH_LOCAL_LOCATION=$ASSETS_DIR/$GS_TT_FLASH_FILENAME
  WH_TT_FLASH_LOCAL_LOCATION=$ASSETS_DIR/$WH_TT_FLASH_FILENAME
  TT_DRIVER_LOCAL_LOCATION=$ASSETS_DIR/$GS_TT_DRIVER_FILENAME
else
  GS_TT_SMI_LOCAL_FOLDER=$ASSETS_DIR/tt_smi/grayskull
  WH_TT_SMI_LOCAL_FOLDER=$ASSETS_DIR/tt_smi/wormhole_b0
  GS_TT_FLASH_LOCAL_FOLDER=$ASSETS_DIR/tt_flash/grayskull
  WH_TT_FLASH_LOCAL_FOLDER=$ASSETS_DIR/tt_flash/wormhole_b0
  TT_DRIVER_LOCAL_FOLDER=$ASSETS_DIR/tt_driver

  mkdir -p $GS_TT_SMI_LOCAL_FOLDER
  mkdir -p $WH_TT_SMI_LOCAL_FOLDER
  mkdir -p $GS_TT_FLASH_LOCAL_FOLDER
  mkdir -p $WH_TT_FLASH_LOCAL_FOLDER
  mkdir -p $TT_DRIVER_LOCAL_FOLDER

  GS_TT_SMI_LOCAL_LOCATION=$GS_TT_SMI_LOCAL_FOLDER/tt-smi
  WH_TT_SMI_LOCAL_LOCATION=$WH_TT_SMI_LOCAL_FOLDER/tt-smi
  GS_TT_FLASH_LOCAL_LOCATION=$GS_TT_FLASH_LOCAL_FOLDER/tt-flash
  WH_TT_FLASH_LOCAL_LOCATION=$WH_TT_FLASH_LOCAL_FOLDER/tt-flash
  TT_DRIVER_LOCAL_LOCATION=$TT_DRIVER_LOCAL_FOLDER/install_ttkmd.bash
fi


echo $GS_TT_SMI_SERVER_LOCATION
echo $GS_TT_SMI_LOCAL_LOCATION
curl -L -H "Authorization: Bearer $GITHUB_TOKEN" \
	-H "X-GitHub-Api-Version: 2022-11-28" \
	-H "Accept: application/octet-stream" \
	$GS_TT_SMI_SERVER_LOCATION -o $GS_TT_SMI_LOCAL_LOCATION

echo $WH_TT_SMI_SERVER_LOCATION
echo $WH_TT_SMI_LOCAL_LOCATION
curl -L -H "Authorization: Bearer $GITHUB_TOKEN" \
	-H "X-GitHub-Api-Version: 2022-11-28" \
	-H "Accept: application/octet-stream" \
	$WH_TT_SMI_SERVER_LOCATION -o $WH_TT_SMI_LOCAL_LOCATION

echo $TT_DRIVER_SERVER_LOCATION
echo $TT_DRIVER_LOCAL_LOCATION
curl -L -H "Authorization: Bearer $GITHUB_TOKEN" \
	-H "X-GitHub-Api-Version: 2022-11-28" \
	-H "Accept: application/octet-stream" \
	$GS_TT_DRIVER_SERVER_LOCATION -o $TT_DRIVER_LOCAL_LOCATION

echo $GS_TT_FLASH_SERVER_LOCATION
echo $GS_TT_FLASH_LOCAL_LOCATION
curl -L -H "Authorization: Bearer $GITHUB_TOKEN" \
	-H "X-GitHub-Api-Version: 2022-11-28" \
	-H "Accept: application/octet-stream" \
	$GS_TT_FLASH_SERVER_LOCATION -o $GS_TT_FLASH_LOCAL_LOCATION

echo $WH_TT_FLASH_SERVER_LOCATION
echo $WH_TT_FLASH_LOCAL_LOCATION
curl -L -H "Authorization: Bearer $GITHUB_TOKEN" \
	-H "X-GitHub-Api-Version: 2022-11-28" \
	-H "Accept: application/octet-stream" \
	$WH_TT_FLASH_SERVER_LOCATION -o $WH_TT_FLASH_LOCAL_LOCATION
