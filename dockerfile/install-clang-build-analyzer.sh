#!/bin/bash
# Install ClangBuildAnalyzer from source
# This script centralizes the version, hash, and installation logic
# to avoid drift between Dockerfile and Dockerfile.fedora
#
# To update ClangBuildAnalyzer version:
# 1. Update CBA_VERSION and CBA_SHA256 below
# 2. Both Dockerfiles will automatically use the new version on next build

set -euo pipefail

# ClangBuildAnalyzer version and SHA256 hash
# Source: https://github.com/aras-p/ClangBuildAnalyzer/releases
CBA_VERSION=1.6.0
CBA_SHA256=868a8d34ecb9b65da4e5874342062a12c081ce4385c7ddd6ce7d557a0c5c292d

# Install ClangBuildAnalyzer
mkdir -p /tmp/cba
wget -q -O /tmp/cba/cba.tar.gz "https://github.com/aras-p/ClangBuildAnalyzer/archive/refs/tags/v${CBA_VERSION}.tar.gz"
echo "${CBA_SHA256}  /tmp/cba/cba.tar.gz" | sha256sum -c -
tar -xzf /tmp/cba/cba.tar.gz -C /tmp/cba --strip-components=1
cmake -S /tmp/cba/ -B /tmp/cba/build -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/cba/build
cmake --install /tmp/cba/build
rm -rf /tmp/cba
