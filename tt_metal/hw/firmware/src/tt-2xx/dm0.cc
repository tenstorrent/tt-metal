#include "firmware_common.h"
#include "risc_common.h"
#include "risc_attribs.h"

tt_l1_ptr mailboxes_t* const mailboxes = (tt_l1_ptr mailboxes_t*)(MEM_MAILBOX_BASE - 3);

int main() {
    configure_csr();
    // WAYPOINT("I");

    mailboxes->go_messages[0].signal = RUN_MSG_DONE;
    while (1) {
        // WAYPOINT("GW");
    }

    return 0;
}
