# set SFPI release version information

sfpi_url=https://github.com/tenstorrent/sfpi/releases/download

# convert md5 file into these variables
# sed 's/^\([0-9a-f]*\) \*sfpi_\([-.0-9a-zA-Z]*\)_\([a-z0-9_A-Z]*\)\.\([a-z]*\)$/sfpi_version=\2'"\n"'sfpi_\3_\4_md5=\1/' *-build/sfpi_*.md5 | sort -u
sfpi_version=7.1.0-bits-18813
sfpi_x86_64_deb_md5=88e017bbb681cecbc1fce10aba805ccb
sfpi_x86_64_rpm_md5=55bf2d3518bcb5d4cd2933ee78f3fd58
sfpi_x86_64_txz_md5=453a99d73ea48e9f4b13bcdf1aea96c4
