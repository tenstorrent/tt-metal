# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Multilingual text cleaning (coqui_cleaners.py, vendored) and the language gate in frontend.py.

The test VECTORS below are coqui's own, lifted from the self-tests in the file the cleaners came
from — the point of vendoring rather than transcribing is that upstream's expectations come with
it. Cases upstream had already commented out as broken are left out.

Czech is absent on purpose: upstream asks num2words for "cz", which does not exist, and there are
no Czech ordinals in num2words either. See frontend.CLEANED_LANGUAGES.
"""
import os

import pytest

from models.experimental.xtts_v2.reference.coqui.cleaners import (
    expand_abbreviations_multilingual,
    expand_numbers_multilingual,
    expand_symbols_multilingual,
)
from models.experimental.xtts_v2.frontend import (
    BASIC_LANGUAGES,
    CLEANED_LANGUAGES,
    NEEDS_PACKAGE,
    SUPPORTED_LANGUAGES,
    XttsTokenizer,
)


@pytest.fixture(scope="module")
def xtts_vocab():
    """vocab.json sits next to the checkpoint, which is where XttsV2 looks for it."""
    from models.experimental.xtts_v2.reference.xtts_gpt_ref import resolve_ckpt

    return os.path.join(os.path.dirname(resolve_ckpt()), "vocab.json")


# --- upstream's vectors, minus the languages we do not route ------------------------------
# Vectors whose expectation came from an older num2words. Marked strict-xfail rather than deleted:
# if num2words changes again these turn XPASS and ask to be looked at.
STALE_VECTORS = {
    ("Через 12.5 секунды.", "ru"): "num2words 0.5.14 renders this as 'двенадцать целых пять десятых'",
}


def _vectors(cases):
    out = []
    for case in cases:
        if case[-1] not in SUPPORTED_LANGUAGES:
            continue
        reason = STALE_VECTORS.get((case[0], case[-1]))
        marks = [pytest.mark.xfail(strict=True, reason=reason)] if reason else []
        out.append(pytest.param(*case, marks=marks))
    return out


NUMBER_test_cases = [
    # English
    ("In 12.5 seconds.", "In twelve point five seconds.", "en"),
    ("There were 50 soldiers.", "There were fifty soldiers.", "en"),
    ("This is a 1st test", "This is a first test", "en"),
    ("That will be $20 sir.", "That will be twenty dollars sir.", "en"),
    ("That will be 20€ sir.", "That will be twenty euro sir.", "en"),
    ("That will be 20.15€ sir.", "That will be twenty euro, fifteen cents sir.", "en"),
    ("That's 100,000.5.", "That's one hundred thousand point five.", "en"),
    # French
    ("En 12,5 secondes.", "En douze virgule cinq secondes.", "fr"),
    ("Il y avait 50 soldats.", "Il y avait cinquante soldats.", "fr"),
    ("Ceci est un 1er test", "Ceci est un premier test", "fr"),
    ("Cela vous fera $20 monsieur.", "Cela vous fera vingt dollars monsieur.", "fr"),
    ("Cela vous fera 20€ monsieur.", "Cela vous fera vingt euros monsieur.", "fr"),
    ("Cela vous fera 20,15€ monsieur.", "Cela vous fera vingt euros et quinze centimes monsieur.", "fr"),
    ("Ce sera 100.000,5.", "Ce sera cent mille virgule cinq.", "fr"),
    # German
    ("In 12,5 Sekunden.", "In zwölf Komma fünf Sekunden.", "de"),
    ("Es gab 50 Soldaten.", "Es gab fünfzig Soldaten.", "de"),
    ("Dies ist ein 1. Test", "Dies ist ein erste Test", "de"),  # Issue with gender
    ("Das macht $20 Herr.", "Das macht zwanzig Dollar Herr.", "de"),
    ("Das macht 20€ Herr.", "Das macht zwanzig Euro Herr.", "de"),
    ("Das macht 20,15€ Herr.", "Das macht zwanzig Euro und fünfzehn Cent Herr.", "de"),
    # Spanish
    ("En 12,5 segundos.", "En doce punto cinco segundos.", "es"),
    ("Había 50 soldados.", "Había cincuenta soldados.", "es"),
    ("Este es un 1er test", "Este es un primero test", "es"),
    ("Eso le costará $20 señor.", "Eso le costará veinte dólares señor.", "es"),
    ("Eso le costará 20€ señor.", "Eso le costará veinte euros señor.", "es"),
    ("Eso le costará 20,15€ señor.", "Eso le costará veinte euros con quince céntimos señor.", "es"),
    # Italian
    ("In 12,5 secondi.", "In dodici virgola cinque secondi.", "it"),
    ("C'erano 50 soldati.", "C'erano cinquanta soldati.", "it"),
    ("Questo è un 1° test", "Questo è un primo test", "it"),
    ("Ti costerà $20 signore.", "Ti costerà venti dollari signore.", "it"),
    ("Ti costerà 20€ signore.", "Ti costerà venti euro signore.", "it"),
    ("Ti costerà 20,15€ signore.", "Ti costerà venti euro e quindici centesimi signore.", "it"),
    # Portuguese
    ("Em 12,5 segundos.", "Em doze vírgula cinco segundos.", "pt"),
    ("Havia 50 soldados.", "Havia cinquenta soldados.", "pt"),
    ("Este é um 1º teste", "Este é um primeiro teste", "pt"),
    ("Isso custará $20 senhor.", "Isso custará vinte dólares senhor.", "pt"),
    ("Isso custará 20€ senhor.", "Isso custará vinte euros senhor.", "pt"),
    (
        "Isso custará 20,15€ senhor.",
        "Isso custará vinte euros e quinze cêntimos senhor.",
        "pt",
    ),  # "cêntimos" should be "centavos" num2words issue
    # Polish
    ("W 12,5 sekundy.", "W dwanaście przecinek pięć sekundy.", "pl"),
    ("Było 50 żołnierzy.", "Było pięćdziesiąt żołnierzy.", "pl"),
    ("To będzie kosztować 20€ panie.", "To będzie kosztować dwadzieścia euro panie.", "pl"),
    ("To będzie kosztować 20,15€ panie.", "To będzie kosztować dwadzieścia euro, piętnaście centów panie.", "pl"),
    # Arabic
    ("في الـ 12,5 ثانية.", "في الـ اثنا عشر  , خمسون ثانية.", "ar"),
    ("كان هناك 50 جنديًا.", "كان هناك خمسون جنديًا.", "ar"),
    # Czech
    ("Za 12,5 vteřiny.", "Za dvanáct celá pět vteřiny.", "cs"),
    ("Bylo tam 50 vojáků.", "Bylo tam padesát vojáků.", "cs"),
    ("To bude stát 20€ pane.", "To bude stát dvacet euro pane.", "cs"),
    ("To bude 20.15€ pane.", "To bude dvacet euro, patnáct centů pane.", "cs"),
    # Russian
    ("Через 12.5 секунды.", "Через двенадцать запятая пять секунды.", "ru"),
    ("Там было 50 солдат.", "Там было пятьдесят солдат.", "ru"),
    ("Это будет 20.15€ сэр.", "Это будет двадцать евро, пятнадцать центов сэр.", "ru"),
    ("Это будет стоить 20€ господин.", "Это будет стоить двадцать евро господин.", "ru"),
    # Dutch
    ("In 12,5 seconden.", "In twaalf komma vijf seconden.", "nl"),
    ("Er waren 50 soldaten.", "Er waren vijftig soldaten.", "nl"),
    ("Dat wordt dan $20 meneer.", "Dat wordt dan twintig dollar meneer.", "nl"),
    ("Dat wordt dan 20€ meneer.", "Dat wordt dan twintig euro meneer.", "nl"),
    # Chinese (Simplified)
    ("在12.5秒内", "在十二点五秒内", "zh"),
    ("有50名士兵", "有五十名士兵", "zh"),
    # Turkish
    ("50 asker vardı.", "elli asker vardı.", "tr"),
    ("Bu 1. test", "Bu birinci test", "tr"),
    # Hungarian
    ("12,5 másodperc alatt.", "tizenkettő egész öt tized másodperc alatt.", "hu"),
    ("50 katona volt.", "ötven katona volt.", "hu"),
    ("Ez az 1. teszt", "Ez az első teszt", "hu"),
    # Korean
    ("12.5 초 안에.", "십이 점 다섯 초 안에.", "ko"),
    ("50 명의 병사가 있었다.", "오십 명의 병사가 있었다.", "ko"),
    ("이것은 1 번째 테스트입니다", "이것은 첫 번째 테스트입니다", "ko"),
]

ABBREV_test_cases = [
    # English
    ("Hello Mr. Smith.", "Hello mister Smith.", "en"),
    ("Dr. Jones is here.", "doctor Jones is here.", "en"),
    # Spanish
    ("Hola Sr. Garcia.", "Hola señor Garcia.", "es"),
    ("La Dra. Martinez es muy buena.", "La doctora Martinez es muy buena.", "es"),
    # French
    ("Bonjour Mr. Dupond.", "Bonjour monsieur Dupond.", "fr"),
    ("Mme. Moreau est absente aujourd'hui.", "madame Moreau est absente aujourd'hui.", "fr"),
    # German
    ("Frau Dr. Müller ist sehr klug.", "Frau doktor Müller ist sehr klug.", "de"),
    # Portuguese
    ("Olá Sr. Silva.", "Olá senhor Silva.", "pt"),
    ("Dra. Costa, você está disponível?", "doutora Costa, você está disponível?", "pt"),
    # Italian
    ("Buongiorno, Sig. Rossi.", "Buongiorno, signore Rossi.", "it"),
    # Polish
    ("Dzień dobry, P. Kowalski.", "Dzień dobry, pani Kowalski.", "pl"),
    ("M. Nowak, czy mogę zadać pytanie?", "pan Nowak, czy mogę zadać pytanie?", "pl"),
    # Czech
    ("P. Novák", "pan Novák", "cs"),
    ("Dr. Vojtěch", "doktor Vojtěch", "cs"),
    # Dutch
    ("Dhr. Jansen", "de heer Jansen", "nl"),
    ("Mevr. de Vries", "mevrouw de Vries", "nl"),
    # Russian
    ("Здравствуйте Г-н Иванов.", "Здравствуйте господин Иванов.", "ru"),
    ("Д-р Смирнов здесь, чтобы увидеть вас.", "доктор Смирнов здесь, чтобы увидеть вас.", "ru"),
    # Turkish
    ("Merhaba B. Yılmaz.", "Merhaba bay Yılmaz.", "tr"),
    ("Dr. Ayşe burada.", "doktor Ayşe burada.", "tr"),
    # Hungarian
    ("Dr. Szabó itt van.", "doktor Szabó itt van.", "hu"),
]

SYMBOL_test_cases = [
    ("I have 14% battery", "I have 14 percent battery", "en"),
    ("Te veo @ la fiesta", "Te veo arroba la fiesta", "es"),
    ("J'ai 14° de fièvre", "J'ai 14 degrés de fièvre", "fr"),
    ("Die Rechnung beträgt £ 20", "Die Rechnung beträgt pfund 20", "de"),
    ("O meu email é ana&joao@gmail.com", "O meu email é ana e joao arroba gmail.com", "pt"),
    ("linguaggio di programmazione C#", "linguaggio di programmazione C cancelletto", "it"),
    ("Moja temperatura to 36.6°", "Moja temperatura to 36.6 stopnie", "pl"),
    ("Mám 14% baterie", "Mám 14 procento baterie", "cs"),
    ("Těším se na tebe @ party", "Těším se na tebe na party", "cs"),
    ("У меня 14% заряда", "У меня 14 процентов заряда", "ru"),
    ("Я буду @ дома", "Я буду собака дома", "ru"),
    ("Ik heb 14% batterij", "Ik heb 14 procent batterij", "nl"),
    ("Ik zie je @ het feest", "Ik zie je bij het feest", "nl"),
    ("لدي 14% في البطارية", "لدي 14 في المئة في البطارية", "ar"),
    ("我的电量为 14%", "我的电量为 14 百分之", "zh"),
    ("Pilim %14 dolu.", "Pilim yüzde 14 dolu.", "tr"),
    ("Az akkumulátorom töltöttsége 14%", "Az akkumulátorom töltöttsége 14 százalék", "hu"),
    ("배터리 잔량이 14%입니다.", "배터리 잔량이 14 퍼센트입니다.", "ko"),
]


@pytest.mark.parametrize("text,expected,lang", _vectors(NUMBER_test_cases))
def test_expand_numbers_multilingual(text, expected, lang):
    assert expand_numbers_multilingual(text, lang) == expected


@pytest.mark.parametrize("text,expected,lang", _vectors(ABBREV_test_cases))
def test_expand_abbreviations_multilingual(text, expected, lang):
    assert expand_abbreviations_multilingual(text, lang) == expected


@pytest.mark.parametrize("text,expected,lang", _vectors(SYMBOL_test_cases))
def test_expand_symbols_multilingual(text, expected, lang):
    assert expand_symbols_multilingual(text, lang) == expected


# --- the language gate in frontend.XttsTokenizer -------------------------------------------
# One sentence per supported language, each carrying a number and an ordinal so the cleaner's
# num2words paths are exercised rather than just the BPE.
GOLDEN_SENTENCES = {
    "en": "The 3rd item costs $5.50.",
    "es": "El 3º artículo cuesta 5,50€.",
    "fr": "Le 3e article coûte 5,50€.",
    "de": "Der 3. Artikel kostet 5,50€.",
    "it": "Il 3º articolo costa 5,50€.",
    "pt": "O 3º item custa 5,50€.",
    "pl": "3. przedmiot kosztuje 5,50€.",
    "tr": "3. ürün 5,50€ tutuyor.",
    "ru": "3-й товар стоит 5,50€.",
    "nl": "Het 3de artikel kost 5,50€.",
    "ar": "العنصر 3 يكلف 5 دولار.",
    "hu": "A 3. tétel 5,50€-ba kerül.",
    "hi": "तीसरी वस्तु की कीमत 5 डॉलर है।",
}


# Recorded output of the sentences above. Upstream's vectors say the cleaned TEXT is right; these
# pin the ids that follow from it, so any drift in a table, in num2words or in vocab.json fails
# here and names the language.
GOLDEN_IDS = {
    "ar": (
        5022,
        4001,
        4106,
        3961,
        3957,
        2,
        3951,
        4017,
        3951,
        3949,
        2,
        3977,
        4096,
        3968,
        2,
        3954,
        4291,
        3949,
        2,
        3955,
        4264,
        3957,
        9,
    ),
    "de": (
        260,
        234,
        2,
        481,
        5183,
        2,
        14,
        5225,
        650,
        249,
        2,
        373,
        368,
        33,
        2,
        427,
        27,
        19,
        2,
        5764,
        121,
        28,
        2,
        320,
        2,
        427,
        27,
        19,
        1476,
        20,
        2,
        5762,
        86,
        9,
    ),
    "en": (
        259,
        42,
        2,
        40,
        98,
        17,
        2,
        60,
        225,
        2,
        1169,
        192,
        2,
        840,
        76,
        2,
        134,
        84,
        59,
        32,
        7,
        2,
        19,
        140,
        210,
        2,
        1217,
        192,
        9,
    ),
    "es": (
        284,
        249,
        2,
        114,
        117,
        89,
        2,
        59,
        3073,
        1840,
        28,
        2,
        825,
        18,
        546,
        2,
        1460,
        174,
        2,
        759,
        1224,
        2,
        135,
        2,
        1460,
        1389,
        434,
        2,
        952,
        27,
        78,
        1145,
        9,
    ),
    "fr": (
        262,
        64,
        2,
        685,
        2215,
        269,
        80,
        2,
        14,
        5225,
        22,
        1095,
        18,
        2,
        174,
        1069,
        18,
        2,
        1460,
        30,
        2,
        759,
        1224,
        2,
        453,
        2,
        1460,
        1583,
        533,
        2,
        1217,
        78,
        80,
        32,
        9,
    ),
    "hu": (
        5753,
        14,
        2,
        1838,
        26,
        68,
        650,
        2,
        811,
        589,
        2,
        5572,
        2,
        806,
        28,
        7,
        2,
        5572,
        200,
        2,
        1217,
        192,
        8,
        1159,
        2,
        93,
        406,
        25,
        9,
    ),
    "it": (
        285,
        151,
        2,
        136,
        2338,
        28,
        2,
        14,
        5225,
        91,
        392,
        28,
        2,
        1169,
        434,
        2,
        1460,
        754,
        2,
        806,
        28,
        2,
        18,
        2,
        1460,
        1583,
        27,
        434,
        2,
        1217,
        699,
        2280,
        9,
    ),
    "nl": (
        297,
        62,
        33,
        2,
        234,
        126,
        2,
        14,
        5225,
        650,
        249,
        2,
        373,
        63,
        2,
        35,
        3021,
        2,
        806,
        28,
        7,
        2,
        35,
        2736,
        384,
        343,
        2,
        1217,
        33,
        9,
    ),
    "pl": (
        294,
        2450,
        9,
        2,
        2460,
        26,
        1185,
        33,
        2,
        373,
        2339,
        803,
        386,
        2,
        2533,
        292,
        2,
        806,
        28,
        7,
        2,
        2533,
        292,
        2353,
        1679,
        290,
        33,
        2,
        1217,
        2629,
        9,
    ),
    "pt": (
        286,
        28,
        2,
        114,
        16,
        1834,
        2,
        60,
        225,
        2,
        825,
        546,
        2,
        1460,
        174,
        2,
        759,
        1224,
        2,
        18,
        2,
        1460,
        754,
        27,
        434,
        2,
        1760,
        27,
        78,
        1145,
        9,
    ),
    "ru": (
        267,
        3392,
        3671,
        3382,
        3383,
        2,
        3414,
        3421,
        3390,
        2,
        3481,
        3426,
        2,
        3758,
        3439,
        2,
        3805,
        3429,
        7,
        2,
        3758,
        3439,
        3504,
        3438,
        3392,
        2,
        3396,
        3410,
        3761,
        9,
    ),
    "tr": (295, 2037, 584, 16, 258, 2, 258, 406, 27, 2, 15, 2277, 7, 18, 84, 22, 1, 2, 803, 33, 2292, 57, 9),
    "hi": (
        6680,
        6299,
        6211,
        6320,
        2,
        6208,
        6274,
        6345,
        2,
        6269,
        2,
        6269,
        6627,
        2,
        4500,
        2,
        6189,
        6225,
        6206,
        6204,
        2,
        6261,
        6241,
    ),
}


def test_every_language_has_a_golden_sentence():
    """A language in SUPPORTED_LANGUAGES with no test sentence would go unexercised."""
    assert sorted(GOLDEN_SENTENCES) == sorted(SUPPORTED_LANGUAGES)


@pytest.mark.parametrize("lang", SUPPORTED_LANGUAGES)
def test_encode_supported_language(lang, xtts_vocab):
    ids = XttsTokenizer(xtts_vocab).encode(GOLDEN_SENTENCES[lang], lang)
    assert len(ids) > 3, f"{lang}: {len(ids)} tokens is too few to be a real encoding"
    assert XttsTokenizer(xtts_vocab).decode(ids).startswith(f"[{lang}]"), "language tag missing"


@pytest.mark.parametrize("lang", sorted(NEEDS_PACKAGE))
def test_language_needing_a_package_is_refused_by_name(lang, xtts_vocab, expect_error):
    """Refuse loudly and name the package: silently mis-cleaning gives plausible-but-wrong audio."""
    package = NEEDS_PACKAGE[lang].split()[0].rstrip(",")  # "pypinyin," -> "pypinyin"
    with expect_error(NotImplementedError, package):
        XttsTokenizer(xtts_vocab).encode("test", lang)


def test_unknown_language_is_refused(xtts_vocab, expect_error):
    with expect_error(NotImplementedError, "not one of"):
        XttsTokenizer(xtts_vocab).encode("test", "xx")


def test_region_suffix_is_stripped(xtts_vocab):
    """ "pt-br" must clean as pt rather than fall through to the unknown-language error."""
    tk = XttsTokenizer(xtts_vocab)
    assert tk.encode(GOLDEN_SENTENCES["pt"], "pt-br") == tk.encode(GOLDEN_SENTENCES["pt"], "pt")


def test_cleaned_and_basic_languages_do_not_overlap():
    assert not set(CLEANED_LANGUAGES) & set(BASIC_LANGUAGES)


@pytest.mark.parametrize("lang", SUPPORTED_LANGUAGES)
def test_token_ids_match_the_golden(lang, xtts_vocab):
    assert tuple(XttsTokenizer(xtts_vocab).encode(GOLDEN_SENTENCES[lang], lang)) == GOLDEN_IDS[lang]


@pytest.mark.parametrize("lang", SUPPORTED_LANGUAGES)
def test_language_tag_is_one_vocab_token(lang, xtts_vocab):
    """A tag the vocab lacks does not raise — it shatters into <unk> characters, so the model gets
    no language instruction and speaks with the wrong phonetics."""
    raw = XttsTokenizer(xtts_vocab).tokenizer
    assert raw.token_to_id(f"[{lang}]") is not None, f"[{lang}] is not in vocab.json"
    assert len(raw.encode(f"[{lang}]").ids) == 1, f"[{lang}] does not encode as a single token"
