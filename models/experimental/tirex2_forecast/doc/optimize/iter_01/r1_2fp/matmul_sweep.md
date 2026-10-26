# matmul sweep (ffn_down,mlstm_qkvo,mlstm_qkv,mlstm_o_out; M = 131072, HiFi4, fp32 acc; median of 10)

| role | config | ms | pcc |
|---|---|---|---|
| mlstm_qkv | linear default | 5.67 | 1.0 |
| mlstm_qkv | minimal M8 K8 N8 sub2x2 | 2.57 | 0.9999999997600746 |
| mlstm_qkv | minimal M8 K8 N8 sub1x4 | 2.61 | 0.9999999997600746 |
| mlstm_qkv | minimal M8 K8 N8 sub4x1 | 2.47 | 0.9999999997600746 |
| mlstm_qkv | minimal M8 K16 N8 sub2x2 | nan | TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| mlstm_qkv | minimal M8 K16 N8 sub1x4 | nan | TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| mlstm_qkv | minimal M8 K16 N8 sub4x1 | nan | TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| mlstm_qkv | minimal M16 K8 N8 sub2x2 | nan | TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| mlstm_qkv | minimal M16 K8 N8 sub1x4 | nan | TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| mlstm_qkv | minimal M16 K8 N8 sub4x1 | nan | TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| mlstm_qkv | minimal M8 K8 N4 sub2x2 | 2.63 | 0.9999999997600746 |
| mlstm_qkv | minimal M8 K8 N4 sub1x4 | 2.57 | 0.9999999997600746 |
| mlstm_qkv | minimal M8 K8 N4 sub4x1 | 2.58 | 0.9999999997600746 |
| mlstm_qkv | minimal M4 K8 N8 sub2x2 | 2.86 | 0.9999999997600746 |
| mlstm_qkv | minimal M4 K8 N8 sub1x4 | 2.82 | 0.9999999997600746 |
| mlstm_qkv | minimal M4 K8 N8 sub4x1 | 2.82 | 0.9999999997600746 |
| mlstm_qkv | minimal M16 K16 N4 sub2x2 | nan | TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| mlstm_qkv | minimal M16 K16 N4 sub1x4 | nan | TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| mlstm_qkv | minimal M16 K16 N4 sub4x1 | nan | TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| mlstm_qkv | minimal M8 K16 N16 sub2x2 | nan | TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| mlstm_qkv | minimal M8 K16 N16 sub1x4 | nan | TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| mlstm_qkv | minimal M8 K16 N16 sub4x1 | nan | TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| mlstm_o_out | linear default | 2.00 | 1.0 |
| mlstm_o_out | minimal M8 K8 N8 sub2x2 | 1.00 | 0.9999999997568747 |
| mlstm_o_out | minimal M8 K8 N8 sub1x4 | 1.00 | 0.9999999997568747 |
| mlstm_o_out | minimal M8 K8 N8 sub4x1 | 1.01 | 0.9999999997568747 |
| mlstm_o_out | minimal M8 K16 N8 sub2x2 | nan | TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| mlstm_o_out | minimal M8 K16 N8 sub1x4 | nan | TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| mlstm_o_out | minimal M8 K16 N8 sub4x1 | nan | TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| mlstm_o_out | minimal M16 K8 N8 sub2x2 | nan | TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| mlstm_o_out | minimal M16 K8 N8 sub1x4 | nan | TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| mlstm_o_out | minimal M16 K8 N8 sub4x1 | nan | TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| mlstm_o_out | minimal M8 K8 N4 sub2x2 | 1.00 | 0.9999999997568747 |
| mlstm_o_out | minimal M8 K8 N4 sub1x4 | 0.99 | 0.9999999997568747 |
| mlstm_o_out | minimal M8 K8 N4 sub4x1 | 1.00 | 0.9999999997568747 |
| mlstm_o_out | minimal M4 K8 N8 sub2x2 | 0.92 | 0.9999999997568747 |
| mlstm_o_out | minimal M4 K8 N8 sub1x4 | 0.92 | 0.9999999997568747 |
| mlstm_o_out | minimal M4 K8 N8 sub4x1 | 0.93 | 0.9999999997568747 |
| mlstm_o_out | minimal M16 K16 N4 sub2x2 | nan | TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| mlstm_o_out | minimal M16 K16 N4 sub1x4 | nan | TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| mlstm_o_out | minimal M16 K16 N4 sub4x1 | nan | TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| mlstm_o_out | minimal M8 K16 N16 sub2x2 | nan | TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| mlstm_o_out | minimal M8 K16 N16 sub1x4 | nan | TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| mlstm_o_out | minimal M8 K16 N16 sub4x1 | nan | TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| ffn_down | linear default | 4.00 | 1.0 |
| ffn_down | minimal M8 K4 N8 sub2x2 | 1.92 | 0.9999999997198917 |
| ffn_down | minimal M8 K4 N8 sub1x4 | 1.88 | 0.9999999997198917 |
| ffn_down | minimal M8 K4 N8 sub4x1 | 1.92 | 0.9999999997198917 |
| ffn_down | minimal M8 K11 N8 sub2x2 | 1.87 | 0.9999999996320237 |
| ffn_down | minimal M8 K11 N8 sub1x4 | 1.86 | 0.9999999996320237 |
| ffn_down | minimal M8 K11 N8 sub4x1 | 1.92 | 0.9999999996320237 |
| ffn_down | minimal M16 K4 N8 sub2x2 | 2.03 | 0.9999999997198917 |
| ffn_down | minimal M16 K4 N8 sub1x4 | 2.05 | 0.9999999997198917 |
| ffn_down | minimal M16 K4 N8 sub4x1 | 2.02 | 0.9999999997198917 |
| ffn_down | minimal M8 K4 N4 sub2x2 | 1.95 | 0.9999999997198917 |
| ffn_down | minimal M8 K4 N4 sub1x4 | 1.95 | 0.9999999997198917 |
| ffn_down | minimal M8 K4 N4 sub4x1 | 1.98 | 0.9999999997198917 |
| ffn_down | minimal M4 K4 N8 sub2x2 | 1.96 | 0.9999999997198917 |
| ffn_down | minimal M4 K4 N8 sub1x4 | 1.98 | 0.9999999997198917 |
| ffn_down | minimal M4 K4 N8 sub4x1 | 1.96 | 0.9999999997198917 |
| ffn_down | minimal M16 K11 N4 sub2x2 | 1.82 | 0.9999999996320237 |
| ffn_down | minimal M16 K11 N4 sub1x4 | 1.86 | 0.9999999996320237 |
| ffn_down | minimal M16 K11 N4 sub4x1 | 1.87 | 0.9999999996320237 |
| ffn_down | minimal M8 K11 N16 sub2x2 | nan | TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| ffn_down | minimal M8 K11 N16 sub1x4 | nan | TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| ffn_down | minimal M8 K11 N16 sub4x1 | nan | TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| mlstm_qkvo | linear default | 7.65 | 1.0 |
| mlstm_qkvo | minimal M8 K8 N8 sub2x2 | 3.55 | 0.9999999997514588 |
| mlstm_qkvo | minimal M8 K8 N8 sub1x4 | 3.46 | 0.9999999997514588 |
| mlstm_qkvo | minimal M8 K8 N8 sub4x1 | 3.47 | 0.9999999997514588 |
| mlstm_qkvo | minimal M8 K16 N8 sub2x2 | nan | TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| mlstm_qkvo | minimal M8 K16 N8 sub1x4 | nan | TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| mlstm_qkvo | minimal M8 K16 N8 sub4x1 | nan | TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| mlstm_qkvo | minimal M16 K8 N8 sub2x2 | nan | TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| mlstm_qkvo | minimal M16 K8 N8 sub1x4 | nan | TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| mlstm_qkvo | minimal M16 K8 N8 sub4x1 | nan | TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| mlstm_qkvo | minimal M8 K8 N4 sub2x2 | 3.43 | 0.9999999997514588 |
| mlstm_qkvo | minimal M8 K8 N4 sub1x4 | 3.19 | 0.9999999997514588 |
| mlstm_qkvo | minimal M8 K8 N4 sub4x1 | 3.21 | 0.9999999997514588 |
| mlstm_qkvo | minimal M4 K8 N8 sub2x2 | 4.17 | 0.9999999997514588 |
| mlstm_qkvo | minimal M4 K8 N8 sub1x4 | 4.09 | 0.9999999997514588 |
| mlstm_qkvo | minimal M4 K8 N8 sub4x1 | 4.19 | 0.9999999997514588 |
| mlstm_qkvo | minimal M16 K16 N4 sub2x2 | nan | TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| mlstm_qkvo | minimal M16 K16 N4 sub1x4 | nan | TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| mlstm_qkvo | minimal M16 K16 N4 sub4x1 | nan | TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| mlstm_qkvo | minimal M8 K16 N16 sub2x2 | nan | TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| mlstm_qkvo | minimal M8 K16 N16 sub1x4 | nan | TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| mlstm_qkvo | minimal M8 K16 N16 sub4x1 | nan | TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
