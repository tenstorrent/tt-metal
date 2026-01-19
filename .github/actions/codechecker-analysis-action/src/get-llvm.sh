#!/bin/bash
set -e
if [[ ! -z "$CODECHECKER_ACTION_DEBUG" ]]; then
  set -x
fi
set -u

echo "::group::Installing LLVM"

update-alternatives --query clang
update-alternatives --query clang-tidy

export DISTRO_FANCYNAME="$(lsb_release -c | awk '{ print $2 }')"
curl -sL http://apt.llvm.org/llvm-snapshot.gpg.key | sudo apt-key add -

if [[ "$IN_LLVM_VERSION" == "latest" ]]; then
  sudo add-apt-repository -y "deb http://apt.llvm.org/$DISTRO_FANCYNAME/ llvm-toolchain-$DISTRO_FANCYNAME main"
  # Get the largest Clang package number available.
  export LLVM_VER="$(apt-cache search --full 'clang-[[:digit:]]*$' | grep '^Package: clang' | cut -d ' ' -f 2 | sort -V | tail -n 1 | sed 's/clang-//')"
else
  sudo add-apt-repository -y "deb http://apt.llvm.org/$DISTRO_FANCYNAME/ llvm-toolchain-$DISTRO_FANCYNAME-$IN_LLVM_VERSION main"
  export LLVM_VER="$IN_LLVM_VERSION"
fi

sudo apt-get -y --no-install-recommends install \
  clang-$LLVM_VER      \
  clang-tidy-$LLVM_VER
sudo update-alternatives --install \
  /usr/bin/clang clang /usr/bin/clang-$LLVM_VER 10000
sudo update-alternatives --install \
  /usr/bin/clang-tidy clang-tidy /usr/bin/clang-tidy-$LLVM_VER 10000

update-alternatives --query clang
update-alternatives --query clang-tidy
echo "::endgroup::"

echo "REAL_VERSION=$(clang --version | head -n 1 | cut -d' ' -f4-)" >> "$GITHUB_OUTPUT"
