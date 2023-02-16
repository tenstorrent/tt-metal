//
// Created by Emilio Munoz on 2019-04-26.
//

#ifndef TENSIX_FW_RAND_H
#define TENSIX_FW_RAND_H

#define BRISC_DATA_STREAM   0x55AAFFAA
#define TRISC0_DATA_STREAM  0xFFAAFFAA
#define TRISC1_DATA_STREAM  0xFFAA5555
#define TRISC2_DATA_STREAM  0xFF555555

//#define BRISC_DATA_STREAM_TX   0x11223344
//#define TRISC0_DATA_STREAM_TX  0x55667788
//#define TRISC1_DATA_STREAM_TX  0x99AABBCC
//#define TRISC2_DATA_STREAM_TX  0xDDEEFF00

//extern volatile uint * mailbox_base_actual[4];

volatile uint * mailbox_base_actual[4] = {
  reinterpret_cast<volatile uint *>(TENSIX_MAILBOX0_BASE),
  reinterpret_cast<volatile uint *>(TENSIX_MAILBOX1_BASE),
  reinterpret_cast<volatile uint *>(TENSIX_MAILBOX2_BASE),
  reinterpret_cast<volatile uint *>(TENSIX_MAILBOX3_BASE) };



//using namespace std;
//using namespace ckernel;

unsigned int shift_lfsr(unsigned int v)
{
  /*
          config          : galois
          length          : 32
          taps            : (30, 20, 4)
          shift-amount    : 1
          shift-direction : right
  */
  enum {
      length = 32,
      tap_0  = 30,
      tap_1  = 20,
      tap_2  =  4
  };
  typedef unsigned int T;
  const T zero = (T)(0);
  const T lsb = zero + (T)(1);
  const T feedback = (
    (lsb << (tap_0 - 1)) ^
    (lsb << (tap_1 - 1)) ^
    (lsb << (tap_2 - 1))
  );
  v = (v >> 1) ^ ((zero - (v & lsb)) & feedback);
  return v;
}

inline void mailbox_write_full_fw(const uint8_t thread, const uint32_t data){
  mailbox_base_actual[thread][0] = data;
}

// Blocking read
inline uint32_t mailbox_read_full_fw(const uint8_t thread){
  return mailbox_base_actual[thread][0];
}

inline bool mailbox_not_empty_full_fw(const uint8_t thread) {
  return mailbox_base_actual[thread][1] > 0;
}


void read_from_mailbox(uint8_t m, uint32_t *rec, uint32_t *dt, bool is_brisc , uint32_t THREAD_ID) {

  uint32_t data_from_trisc;
  uint8_t transmitter;
  uint32_t exp_data_from_trisc;

  if (is_brisc)
    data_from_trisc = mailbox_read_full_fw(m) ;
  else
    data_from_trisc = mailbox_read_full_fw(m);

  // where does it comes from :
  transmitter = data_from_trisc >> 24;

  //FWLOG1("BEFORE trans: %0d", rec[transmitter]) ;
  rec[transmitter]++;
  //FWLOG1("AFTER trans: %0d", rec[transmitter]) ;

  if (transmitter != 0 && transmitter != 1 &&  transmitter != 2 && transmitter != 3){
    FWASSERT("MAILBOX TEST BRISC FW: NO SUCH THREAD!!!! from received data", 0);
  }
  dt[transmitter] = shift_lfsr(dt[transmitter]);
  exp_data_from_trisc = dt[transmitter] & 0x00FFFFFF;
  if (exp_data_from_trisc != (data_from_trisc & 0x00FFFFFF)){

    if (is_brisc){
      FWLOG2("BRISC exp: %0x act: %0x", exp_data_from_trisc, data_from_trisc);
    } else {

      switch (THREAD_ID){
        case 0:
          FWLOG2("TRISC0 exp: %0x act: %0x", exp_data_from_trisc, data_from_trisc);
          break;
        case 1:
          FWLOG2("TRISC1 exp: %0x act: %0x", exp_data_from_trisc, data_from_trisc);
          break;
        default:
          FWLOG2("TRISC2 exp: %0x act: %0x", exp_data_from_trisc, data_from_trisc);
          break;
      }

    }

    FWASSERT("MAILBOX TEST BRISC FW: ERROR MISMATCH! on data see above", 0);
  } else {
    if (is_brisc) {
      FWLOG2("BRISC FW: MATCH! mailbox[%0d] = %0x", m, data_from_trisc);
      //FWLOG1("BRISC FW: rec[0] = %0d", rec[0]);
      //FWLOG1("BRISC FW: rec[1] = %0d", rec[1]);
      //FWLOG1("BRISC FW: rec[2] = %0d", rec[2]);
      //FWLOG1("BRISC FW: rec[3] = %0d", rec[3]);
    }else {

      switch (THREAD_ID){
        case 0:
          FWLOG2("TRISC0 FW: MATCH! mailbox[%0d] = %0x", m, data_from_trisc);
          //FWLOG1("TRISC0 FW: rec[0] = %0d", rec[0]);
          //FWLOG1("TRISC0 FW: rec[1] = %0d", rec[1]);
          //FWLOG1("TRISC0 FW: rec[2] = %0d", rec[2]);
          //FWLOG1("TRISC0 FW: rec[3] = %0d", rec[3]);
          break;
        case 1:
          FWLOG2("TRISC1 FW: MATCH! mailbox[%0d] = %0x", m, data_from_trisc);
          //FWLOG1("TRISC1 FW: rec[0] = %0d", rec[0]);
          //FWLOG1("TRISC1 FW: rec[1] = %0d", rec[1]);
          //FWLOG1("TRISC1 FW: rec[2] = %0d", rec[2]);
          //FWLOG1("TRISC1 FW: rec[3] = %0d", rec[3]);
          break;
        default:
          FWLOG2("TRISC2 FW: MATCH! mailbox[%0d] = %0x", m, data_from_trisc);
          //FWLOG1("TRISC2 FW: rec[0] = %0d", rec[0]);
          //FWLOG1("TRISC2 FW: rec[1] = %0d", rec[1]);
          //FWLOG1("TRISC2 FW: rec[2] = %0d", rec[2]);
          //FWLOG1("TRISC2 FW: rec[3] = %0d", rec[3]);
          break;
      }

    }

  }

}



#endif //TENSIX_FW_RAND_H
