#!/usr/bin/env bash
#
# Downloads required large files for RVC.

function download() {
  local path="$1"
  echo "Downloading ${path}"
  git lfs pull --include="${path}"
}

set -e

REPO_FOLDER="VoiceConversionWebUI"

assets_commit_hash="$1"
assets_dir="$2"

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "Do not source this script. Run: ./assets-download.sh <commit-hash> <assets-dir>"
  return 1
fi

if [[ -z "${assets_commit_hash}" || -z "${assets_dir}" ]]; then
  echo "Usage: ./assets-download.sh <commit-hash> <assets-dir>"
  exit 1
fi

if [[ "${assets_dir}" == "/" ]]; then
  echo "Refusing to use '/' as assets-dir."
  exit 1
fi

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
  git clone https://huggingface.co/lj1995/VoiceConversionWebUI "${REPO_FOLDER}"
fi

pushd "${REPO_FOLDER}"

git config advice.detachedHead false

git fetch --all --tags
git checkout "${assets_commit_hash}"

unset GIT_LFS_SKIP_SMUDGE
unset GIT_CLONE_PROTECTION_ACTIVE

download "hubert_base.pt"
download "pretrained"
download "uvr5_weights"
download "rmvpe.pt"
download "rmvpe.onnx"

rm -rf .git

popd

mkdir -p "${assets_dir}"

mv "${REPO_FOLDER}/hubert_base.pt" "${assets_dir}/hubert_base.pt"

mkdir -p "${assets_dir}/rmvpe"

mv "${REPO_FOLDER}/rmvpe.pt" "${assets_dir}/rmvpe/rmvpe.pt"
mv "${REPO_FOLDER}/rmvpe.onnx" "${assets_dir}/rmvpe/rmvpe.onnx"

mv "${REPO_FOLDER}/pretrained" "${assets_dir}/pretrained"
mv "${REPO_FOLDER}/uvr5_weights" "${assets_dir}/uvr5_weights"
