# set SFPI release version information

sfpi_url=https://github.com/tenstorrent/sfpi/releases/download

# convert md5 file into these variables
# sed 's/^\([0-9a-f]*\) \*sfpi_\([-.0-9a-zA-Z]*\)_\([a-z0-9_A-Z]*\)\.\([a-z]*\)$/sfpi_version=\2'"\n"'sfpi_\3_\4_md5=\1/' *-build/sfpi_*.md5 | sort -u
sfpi_aarch64_deb_md5=8b2cc93f495b2c44b1ef972dc8cb4c58
sfpi_aarch64_rpm_md5=6ea19152598e34a447cb128c8ab2657c
sfpi_aarch64_txz_md5=1abfa3f32200653b816e7e79d7ec84d7
sfpi_version=7.4.0
sfpi_x86_64_deb_md5=6609998049302ec4e28bf28972df5245
sfpi_x86_64_rpm_md5=26c584b9e8ccdaacdaf560245a47d3f8
sfpi_x86_64_txz_md5=15ed2781ad2cf46c78a8f0310de91f9f
