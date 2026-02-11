#!/usr/bin/env bash
#
# Downloads required large files for RVC.

function download() {
  local path="$1"
  echo "Downloading ${path}"
  git lfs pull --include="${path}"
}

set -e

REPO_FOLDER="rvc-nano"

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

rm -rf .git

popd

mkdir -p "assets"
mkdir -p "configs"

mv "${REPO_FOLDER}/assets" "./"
mv "${REPO_FOLDER}/configs" "./" 


rm -rf "${REPO_FOLDER}"