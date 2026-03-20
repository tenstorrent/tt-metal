ACCEPT = 1
REJECT = 0

BASE = 0
SPEC = 1


def send_to_host(token, token_pos):
    async_forward(
        {
            "token": token,
            "token_pos": token_pos,
        },
        stage="host",
        channel="host_out",
    )


def stage_0():
    while True:
        msg = async_wait_for_inputs(channel="token_in")

        token = msg["token"]
        token_type = msg["token_type"]
        token_pos = msg["token_pos"]

        hidden = run_embedding(token)
        decoder_hidden = run_decoder(hidden, token_pos, token_type)

        async_forward(
            {
                "decoder_hidden": decoder_hidden,
                "source_token": token,
                "source_token_type": token_type,
                "source_token_pos": token_pos,
            },
            stage=stage_1,
            channel="from_stage_0",
        )


def stage_1():
    while True:
        msg = async_wait_for_inputs(channel="from_stage_0")

        decoder_hidden = msg["decoder_hidden"]
        source_token = msg["source_token"]
        source_token_type = msg["source_token_type"]
        source_token_pos = msg["source_token_pos"]

        base_output_token = lm_head_sampling(decoder_hidden)

        e = run_embedding(base_output_token)
        eh_hidden = concat(h_rms_norm(decoder_hidden), e_rms_norm(e))
        mtp_hidden = matmul(eh_hidden, eh_proj)

        async_forward(
            {
                "mtp_hidden": mtp_hidden,
                "source_token": source_token,
                "source_token_type": source_token_type,
                "source_token_pos": source_token_pos,
                "base_output_token": base_output_token,
            },
            stage=stage_2,
            channel="from_stage_1",
        )


def stage_2():
    while True:
        msg = async_wait_for_inputs(channel="from_stage_1")

        mtp_hidden = msg["mtp_hidden"]
        source_token = msg["source_token"]
        source_token_type = msg["source_token_type"]
        source_token_pos = msg["source_token_pos"]
        base_output_token = msg["base_output_token"]

        mtp_decoder_hidden = run_mtp_decoder(
            mtp_hidden,
            source_token_pos,
            source_token_type,
        )

        async_forward(
            {
                "_channel": "mtp_sampling",
                "mtp_decoder_hidden": mtp_decoder_hidden,
                "source_token": source_token,
                "source_token_type": source_token_type,
                "source_token_pos": source_token_pos,
                "base_output_token": base_output_token,
            },
            stage=stage_3,
            channel="mtp_sampling",
        )


def stage_3():
    # speculative token for logical position p that has not yet been verified
    unverified_spec_by_pos = {5: "was"}  # seeded from prefill in this example

    # speculative token for logical position p that has already been accepted
    verified_spec_by_pos = {}

    while True:
        msg = async_wait_for_inputs(channel="mtp_sampling")

        source_token = msg["source_token"]
        source_token_type = msg["source_token_type"]
        source_token_pos = msg["source_token_pos"]
        base_output_token = msg["base_output_token"]
        mtp_hidden_states = msg["mtp_decoder_hidden"]

        verified_pos = source_token_pos + 1
        next_spec_pos = source_token_pos + 2

        # ---------------------------------------------------------
        # BASE packet reached stage 3
        # ---------------------------------------------------------
        if source_token_type == BASE:
            expected_spec = unverified_spec_by_pos.get(verified_pos)

            # ACCEPT: current base output matches saved speculation
            if expected_spec is not None and base_output_token == expected_spec:
                verified_spec_by_pos[verified_pos] = expected_spec
                del unverified_spec_by_pos[verified_pos]

                # accepted speculative token is now committed -> send to host
                send_to_host(expected_spec, verified_pos)

                # do not sample from this BASE packet
                # we want to continue with the already-in-flight SPEC packet
                continue

            # REJECT: current base output does not match saved speculation
            speculative_token = lm_head_sampling(mtp_hidden_states)

            # save new speculation produced from this BASE packet
            unverified_spec_by_pos[next_spec_pos] = speculative_token

            # forward fallback actual token back into pipeline
            async_forward(
                {
                    "token": base_output_token,
                    "token_type": BASE,
                    "token_pos": verified_pos,
                },
                stage=stage_0,
                channel="token_in",
            )

            # fallback actual token is committed -> send to host
            send_to_host(base_output_token, verified_pos)

            # forward new speculative token behind it
            async_forward(
                {
                    "token": speculative_token,
                    "token_type": SPEC,
                    "token_pos": next_spec_pos,
                },
                stage=stage_0,
                channel="token_in",
            )
            continue

        # ---------------------------------------------------------
        # SPEC packet reached stage 3
        # ---------------------------------------------------------
        if source_token_type == SPEC:
            accepted_spec = verified_spec_by_pos.get(source_token_pos)

            # if this SPEC token was never accepted, ignore stale work
            if accepted_spec is None:
                continue

            # this SPEC branch is now the committed continuation
            async_forward(
                {
                    "token": base_output_token,
                    "token_type": BASE,
                    "token_pos": verified_pos,
                },
                stage=stage_0,
                channel="token_in",
            )

            # committed continuation token -> send to host
            send_to_host(base_output_token, verified_pos)

            speculative_token = lm_head_sampling(mtp_hidden_states)
            unverified_spec_by_pos[next_spec_pos] = speculative_token

            # speculative token goes back into pipeline, but NOT to host yet
            async_forward(
                {
                    "token": speculative_token,
                    "token_type": SPEC,
                    "token_pos": next_spec_pos,
                },
                stage=stage_0,
                channel="token_in",
            )

            del verified_spec_by_pos[source_token_pos]
            continue

        raise ValueError("Invalid token type")
