#!/usr/bin/env bash

# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

# Downloads required large files for RVC.

function download() {
  local path="$1"
  echo "Downloading ${path}"
  git lfs pull --include="${path}"
}

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_FOLDER="${SCRIPT_DIR}/rvc-nano"
DATA_DIR="${SCRIPT_DIR}/data"
TARGET_ASSETS_DIR="${DATA_DIR}/assets"
TARGET_CONFIGS_DIR="${DATA_DIR}/configs"
TARGET_SAMPLE_FILE="${DATA_DIR}/sample-speech.wav"

export GIT_CLONE_PROTECTION_ACTIVE=false
export GIT_LFS_SKIP_SMUDGE=1

if [[ -d "${REPO_FOLDER}" ]]; then
  if [[ -d "${REPO_FOLDER}/.git" ]]; then
    echo "Reusing existing ${REPO_FOLDER} clone."
  else
    echo "Removing existing ${REPO_FOLDER} (not a git clone)."
    rm -rf "${REPO_FOLDER}"
  fi
fi

if [[ ! -d "${REPO_FOLDER}" ]]; then
  git clone --depth=1 --no-single-branch https://huggingface.co/mert-kurttutan/rvc-nano "${REPO_FOLDER}"
fi

pushd "${REPO_FOLDER}"

git config advice.detachedHead false

git fetch --depth=1 origin main
git checkout FETCH_HEAD

unset GIT_LFS_SKIP_SMUDGE
unset GIT_CLONE_PROTECTION_ACTIVE

download "assets"
download "configs"
download "sample-speech.wav"

rm -rf .git

popd

mkdir -p "${DATA_DIR}"
rm -rf "${TARGET_ASSETS_DIR}" "${TARGET_CONFIGS_DIR}"
mv "${REPO_FOLDER}/assets" "${TARGET_ASSETS_DIR}"
mv "${REPO_FOLDER}/configs" "${TARGET_CONFIGS_DIR}"
mv "${REPO_FOLDER}/sample-speech.wav" "${TARGET_SAMPLE_FILE}"


rm -rf "${REPO_FOLDER}"
