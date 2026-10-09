"""Fish Audio S2 Pro (fishaudio/s2-pro) text-to-speech on Tenstorrent Blackhole.

Layout: tt/ (TTNN model + generator), reference/ (trimmed torch copies of fish-speech used by tests and
the CPU codec encoder), server/ (FastAPI app served by tt-model's tt-dit-server kind), tests/, demo/.
"""
