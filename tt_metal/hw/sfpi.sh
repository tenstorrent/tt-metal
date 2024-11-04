#! /bin/bash

# Script to automatically get a specified release.  Invoke from your build system providing:
# VERSION file containing desired version
# MD5 file containing md5sum checksum
# DST target directory
# The script is lazy and will not redownload an already-downloaded version

set -eo pipefail

if [[ "$#" != 3 ]] ; then
    echo "Usage: $0 VERSIONFILE MD5FILE DSTDIR" 1>&2
    exit 1
fi

verfile="$1"
md5file="$2"
dstdir="${3%/sfpi}"

if cmp -s "$verfile" "$dstdir/sfpi/sfpi.version" ; then
    # We have this already
    touch "$dstdir/sfpi/sfpi.version"
    echo "Already have sfpi release $(cat "$verfile")"
    exit 0
fi

if which curl >/dev/null ; then
    fetcher="curl -L -o - --ftp-pasv --retry 10"
elif which wget > /dev/null ; then
    fetcher="wget -O -"
else
    echo "No downloader available" 1>&2
    exit 1
fi

url=https://github.com/tenstorrent/sfpi/releases/download
read hash tarball < "$md5file"
read ver < "$verfile"

echo "Downloading new sfpi release: $ver/$tarball"
mkdir -p "$dstdir"
$fetcher "$url/$ver/$tarball" > "$dstdir/$tarball"
if ! (cd "$dstdir" ; md5sum -c -) < "$md5file" ; then
    echo "MD5 hash mismatch on $dstdir/$tarball" 1>&2
    exit 1
fi

(cd "$dstdir" && rm -rf sfpi && tar xzf "$tarball")
echo "$ver" > "$dstdir/sfpi/sfpi.version"
rm -f "$dstdir/$tarball"
