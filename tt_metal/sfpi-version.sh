# set SFPI release version information

sfpi_version=v6.12.0-insn-synth
sfpi_url=https://github.com/tenstorrent/sfpi/releases/download

# convert md5 file into these variables
# sed 's/^\([0-9a-f]*\) \*sfpi-\([a-z0-9_A-Z]*\)\.\([a-z]*\)$/sfpi_\2_\3_md5=\1/'
sfpi_x86_64_Linux_txz_md5=bff2d1a8f3ec5fc600a8dc0f614b486a
sfpi_x86_64_Linux_deb_md5=47493bb0accad9beeb25eef4390f81c9
sfpi_x86_64_Linux_rpm_md5=589d64dc6e3466c2600fd970b438abe5
