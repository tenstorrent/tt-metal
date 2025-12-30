# PI0 Model for Tenstorrent

PI0 (Physical Intelligence Zero) is a vision-language-action model for robotics
that combines a vision encoder, language model, and action expert for end-to-end
robot control.

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                         PI0 Model                              │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐│
│  │   SigLIP        │  │   Gemma 2B      │  │   Gemma 300M    ││
│  │   Vision Tower  │  │   VLM Backbone  │  │   Action Expert ││
│  │   (27 blocks)   │  │   (18 blocks)   │  │   (18 blocks)   ││
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘│
│           │                    │                    │         │
│           └──────────┬─────────┴──────────┬────────┘         │
│                      │                    │                   │
│              ┌───────▼────────┐  ┌───────▼────────┐          │
│              │ Prefix Embed   │  │ Suffix Embed   │          │
│              │ (Images+Lang)  │  │ (State+Action) │          │
│              └───────┬────────┘  └───────┬────────┘          │
│                      │                    │                   │
│                      └──────────┬─────────┘                   │
│                                 │                             │
│                      ┌──────────▼──────────┐                  │
│                      │   Shared Attention   │                  │
│                      └──────────┬──────────┘                  │
│                                 │                             │
│                      ┌──────────▼──────────┐                  │
│                      │  Flow Matching      │                  │
│                      │  Denoiser           │                  │
│                      └──────────┬──────────┘                  │
│                                 │                             │
│                                 ▼                             │
│                         Action Output                         │
└────────────────────────────────────────────────────────────────┘
```
