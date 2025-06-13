# set SFPI release version information

sfpi_version=v6.12.0-insn-synth
sfpi_url=https://github.com/tenstorrent/sfpi/releases/download

# convert md5 file into these variables
# sed 's/^\([0-9a-f]*\) \*sfpi-\([a-z0-9_A-Z]*\)\.\([a-z]*\)$/sfpi_\2_\3_md5=\1/'
sfpi_x86_64_Linux_txz_md5=84537d91516229105e87999f4f6883f4
sfpi_x86_64_Linux_deb_md5=b530886f9b19a26ec3283789235c110e
sfpi_x86_64_Linux_rpm_md5=719c4b0aa8d6e35b1b01c5daf6836293
