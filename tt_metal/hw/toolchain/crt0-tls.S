
	// TLS symbols needed in linker script (one cannot set the type there)

	.macro tls name, section
	.pushsection \section,"awT"
	.global \name
	\name:
	.type \name, @object
	.popsection
	.endm

	tls __ldm_tdata_start, .tdata.start
	tls __ldm_tdata_end, .tdata.end
	tls __ldm_tbss_end, .tbss.end
